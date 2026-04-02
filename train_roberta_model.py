import copy
import os
import random
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

try:
    import psutil
except ImportError:
    psutil = None


RANDOM_STATE = 42

# Faster transformer choice for CPU
MODEL_NAME = "distilroberta-base"

# Speed/quality balance
MAX_LENGTH = 128
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 1.5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2

# Sample sizes per sentiment class for faster CPU training
TRAIN_PER_CLASS = 2500
VAL_PER_CLASS = 500
TEST_PER_CLASS = 500

DATA_DIR = Path("dataset")
TRAIN_PATH = DATA_DIR / "amazon_reviews_train.csv"
VAL_PATH = DATA_DIR / "amazon_reviews_val.csv"
TEST_PATH = DATA_DIR / "amazon_reviews_test.csv"

MODEL_DIR = DATA_DIR / "roberta_sentiment_model_fast"
MODEL_PATH = DATA_DIR / "best_single_review_model_roberta_fast.joblib"
SUMMARY_PATH = DATA_DIR / "best_single_model_roberta_fast_results.csv"
REPORT_PATH = DATA_DIR / "best_single_model_roberta_fast_classification_report.csv"
CONFUSION_PATH = DATA_DIR / "best_single_model_roberta_fast_confusion_matrix.csv"
SEARCH_LOG_PATH = DATA_DIR / "best_single_model_roberta_fast_training_log.csv"


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def mem_gb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def log_step(message):
    mem = mem_gb()
    if mem is None:
        print(f"[{now_str()}] {message}", flush=True)
    else:
        print(f"[{now_str()}] {message} | RAM: {mem:.2f} GB", flush=True)


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds % 60
    return f"{minutes}m {remain:.1f}s"


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_text_column(df):
    candidates = [
        "clean_text",
        "text",
        "review_text",
        "review",
        "content",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a text column. Available columns are: {list(df.columns)}"
    )


def choose_label_column(df):
    candidates = [
        "sentiment",
        "label",
        "sentiment_label",
        "class",
        "target",
        "overall",
        "stars",
        "rating",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a label column. Available columns are: {list(df.columns)}"
    )


def convert_rating_to_sentiment(series):
    s = pd.to_numeric(series, errors="coerce")

    def map_one(x):
        if pd.isna(x):
            return np.nan
        if x <= 2:
            return "negative"
        elif x == 3:
            return "neutral"
        else:
            return "positive"

    return s.apply(map_one)


def build_input_text(df):
    return (
        "Category: "
        + df["category"].fillna("unknown").astype(str)
        + ". Review: "
        + df["clean_text"].fillna("").astype(str)
    ).tolist()


def balanced_sample_by_sentiment(df, per_class, seed=RANDOM_STATE):
    if "sentiment" not in df.columns:
        raise ValueError(f"'sentiment' column not found. Columns: {df.columns.tolist()}")

    parts = []
    for label in sorted(df["sentiment"].dropna().unique()):
        part = df[df["sentiment"] == label].sample(
            n=min((df["sentiment"] == label).sum(), per_class),
            random_state=seed
        )
        parts.append(part)

    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print("Sampled columns:", sampled.columns.tolist())
    return sampled


def compute_metrics(y_true, y_pred, label_names):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=label_names,
        columns=label_names,
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
    }
    return metrics, report, cm


def print_metrics(title, metrics, report, cm):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report:")
    print(report.round(4).to_string())

    print("\nConfusion Matrix:")
    print(cm.to_string())


class ReviewTextDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


class TextClassificationCollator:
    def __init__(self, tokenizer, max_length=MAX_LENGTH):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
        self.max_length = max_length

    def __call__(self, features):
        texts = [f["text"] for f in features]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        rows = []
        for i in range(len(texts)):
            row = {k: encoded[k][i] for k in encoded}
            if "labels" in features[i]:
                row["labels"] = features[i]["labels"]
            rows.append(row)

        return self.data_collator(rows)


class RobertaSentimentPredictor:
    def __init__(self, model_dir, label_encoder, max_length=MAX_LENGTH, batch_size=EVAL_BATCH_SIZE):
        self.model_dir = str(model_dir)
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None
        self._device = None

    def _lazy_load(self):
        if self._tokenizer is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.to(self._device)
            self._model.eval()

    def _to_texts(self, X):
        if isinstance(X, pd.DataFrame):
            work = X.copy()

            if "clean_text" not in work.columns:
                if "text" in work.columns:
                    work["clean_text"] = work["text"].fillna("").astype(str)
                else:
                    raise ValueError("Input DataFrame must contain 'clean_text' or 'text'.")

            if "category" not in work.columns:
                work["category"] = "unknown"

            return build_input_text(work)

        if isinstance(X, (list, tuple, np.ndarray, pd.Series)):
            return [str(x) for x in X]

        raise TypeError("Unsupported input type for predict().")

    @torch.no_grad()
    def predict(self, X):
        self._lazy_load()
        texts = self._to_texts(X)

        dataset = ReviewTextDataset(texts=texts, labels=None)
        collator = TextClassificationCollator(self._tokenizer, max_length=self.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        all_preds = []
        for batch in loader:
            batch = {k: v.to(self._device) for k, v in batch.items()}
            logits = self._model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())

        return np.asarray(all_preds, dtype=int)

    @torch.no_grad()
    def predict_proba(self, X):
        self._lazy_load()
        texts = self._to_texts(X)

        dataset = ReviewTextDataset(texts=texts, labels=None)
        collator = TextClassificationCollator(self._tokenizer, max_length=self.max_length)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        all_probs = []
        for batch in loader:
            batch = {k: v.to(self._device) for k, v in batch.items()}
            logits = self._model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)


def make_class_weights(y_train, num_labels):
    counts = np.bincount(y_train, minlength=num_labels).astype(np.float64)
    weights = counts.sum() / (num_labels * np.maximum(counts, 1.0))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def run_eval(model, loader, device, label_names, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    metrics, report, cm = compute_metrics(
        np.asarray(all_labels, dtype=int),
        np.asarray(all_preds, dtype=int),
        label_names,
    )
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, report, cm


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_items = 0

    for step, batch in enumerate(loader, start=1):
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        if step % 25 == 0:
            print(
                f"[{now_str()}]   batch {step}/{len(loader)} "
                f"- train_loss={total_loss / max(total_items, 1):.4f}",
                flush=True,
            )

    return total_loss / max(total_items, 1)


def main():
    overall_start = time.time()
    set_seed(RANDOM_STATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"Using device: {device}")

    log_step("Step 1/12: Loading train/val/test CSV files")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VAL_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    print("Train columns:", train_df.columns.tolist())
    print("Val columns:", val_df.columns.tolist())
    print("Test columns:", test_df.columns.tolist())

    log_step("Step 2/12: Checking text column")
    text_col = choose_text_column(train_df)
    if text_col != "clean_text":
        log_step(f"clean_text not found, copying from '{text_col}'")
        train_df["clean_text"] = train_df[text_col].fillna("").astype(str)
        val_df["clean_text"] = val_df[text_col].fillna("").astype(str)
        test_df["clean_text"] = test_df[text_col].fillna("").astype(str)

    log_step("Step 3/12: Cleaning text and category columns")
    for df in (train_df, val_df, test_df):
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
        df["category"] = df["category"].fillna("unknown").astype(str)

    log_step("Step 4/12: Finding label column")
    label_col = choose_label_column(train_df)
    log_step(f"Using label column: {label_col}")

    if label_col != "sentiment":
        if label_col in ["overall", "stars", "rating"]:
            log_step(f"Converting {label_col} to sentiment labels")
            train_df["sentiment"] = convert_rating_to_sentiment(train_df[label_col])
            val_df["sentiment"] = convert_rating_to_sentiment(val_df[label_col])
            test_df["sentiment"] = convert_rating_to_sentiment(test_df[label_col])
        else:
            train_df["sentiment"] = train_df[label_col].astype(str)
            val_df["sentiment"] = val_df[label_col].astype(str)
            test_df["sentiment"] = test_df[label_col].astype(str)

    # Remove rows with missing sentiment after conversion
    train_df = train_df.dropna(subset=["sentiment"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["sentiment"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["sentiment"]).reset_index(drop=True)

    log_step("Step 5/12: Downsampling for faster 3-epoch training")
    train_df = balanced_sample_by_sentiment(train_df, TRAIN_PER_CLASS)
    val_df = balanced_sample_by_sentiment(val_df, VAL_PER_CLASS)
    test_df = balanced_sample_by_sentiment(test_df, TEST_PER_CLASS)

    print("After sampling train columns:", train_df.columns.tolist())
    print("After sampling val columns:", val_df.columns.tolist())
    print("After sampling test columns:", test_df.columns.tolist())

    log_step(
        f"Sampled sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    log_step("Step 6/12: Encoding labels")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"].astype(str))
    y_val = label_encoder.transform(val_df["sentiment"].astype(str))
    y_test = label_encoder.transform(test_df["sentiment"].astype(str))
    label_names = list(label_encoder.classes_)
    num_labels = len(label_names)

    log_step("Step 7/12: Building input texts")
    train_texts = build_input_text(train_df)
    val_texts = build_input_text(val_df)
    test_texts = build_input_text(test_df)

    print()
    print(f"[{now_str()}] Model name: {MODEL_NAME}")
    print(f"[{now_str()}] Train/Val/Test sizes: {len(train_texts)}, {len(val_texts)}, {len(test_texts)}")
    print(f"[{now_str()}] Label order: {label_names}")
    print(f"[{now_str()}] Max sequence length: {MAX_LENGTH}")
    print(f"[{now_str()}] Epochs: {NUM_EPOCHS}")
    print(f"[{now_str()}] Train batch size: {TRAIN_BATCH_SIZE}")
    print(f"[{now_str()}] Eval batch size: {EVAL_BATCH_SIZE}")

    log_step("Step 8/12: Loading tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )
    model.to(device)

    train_dataset = ReviewTextDataset(train_texts, y_train)
    val_dataset = ReviewTextDataset(val_texts, y_val)
    test_dataset = ReviewTextDataset(test_texts, y_test)

    collator = TextClassificationCollator(tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    class_weights = make_class_weights(y_train, num_labels).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_train_steps = max(NUM_EPOCHS * len(train_loader), 1)
    warmup_steps = int(WARMUP_RATIO * total_train_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    history_rows = []
    best_state_dict = None
    best_val_metrics = None
    best_epoch = -1
    best_val_score = -1.0
    epochs_without_improve = 0

    log_step("Step 9/12: Starting training")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print("\n" + "=" * 110)
        log_step(f"Epoch {epoch}/{NUM_EPOCHS}: Training")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_fn,
        )
        log_step(f"Epoch {epoch}/{NUM_EPOCHS}: Train loss = {train_loss:.4f}")

        log_step(f"Epoch {epoch}/{NUM_EPOCHS}: Evaluating on validation set")
        val_metrics, val_report, val_cm = run_eval(
            model,
            val_loader,
            device,
            label_names,
            loss_fn,
        )
        val_metrics["train_loss"] = train_loss
        print_metrics(f"Validation | epoch {epoch}", val_metrics, val_report, val_cm)

        row = {"epoch": epoch, "train_loss": train_loss}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["epoch_seconds"] = round(time.time() - epoch_start, 2)
        history_rows.append(row)

        if val_metrics["macro_f1"] > best_val_score:
            best_val_score = val_metrics["macro_f1"]
            best_state_dict = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improve = 0
            log_step(
                f"Epoch {epoch}/{NUM_EPOCHS}: New best model "
                f"(validation macro_f1 = {best_val_score:.4f})"
            )
        else:
            epochs_without_improve += 1
            log_step(
                f"Epoch {epoch}/{NUM_EPOCHS}: No improvement "
                f"(best validation macro_f1 = {best_val_score:.4f})"
            )

        log_step(
            f"Epoch {epoch}/{NUM_EPOCHS}: Total elapsed "
            f"{format_seconds(time.time() - epoch_start)}"
        )

        if epochs_without_improve >= EARLY_STOPPING_PATIENCE:
            log_step("Early stopping triggered")
            break

    log_step("Step 10/12: Loading best checkpoint and evaluating on test set")
    if best_state_dict is None:
        raise RuntimeError("No best model state was saved during training.")

    model.load_state_dict(best_state_dict)

    test_metrics, test_report, test_cm = run_eval(
        model,
        test_loader,
        device,
        label_names,
        loss_fn,
    )
    print_metrics("Final Test Results", test_metrics, test_report, test_cm)

    log_step("Step 11/12: Saving Hugging Face model and bundle")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    predictor = RobertaSentimentPredictor(
        model_dir=MODEL_DIR,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
        batch_size=EVAL_BATCH_SIZE,
    )

    bundle = {
        "model": predictor,
        "label_encoder": label_encoder,
        "label_names": label_names,
        "model_name": MODEL_NAME,
        "model_dir": str(MODEL_DIR),
        "best_epoch": best_epoch,
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    joblib.dump(bundle, MODEL_PATH)
    pd.DataFrame(history_rows).to_csv(SEARCH_LOG_PATH, index=False)
    pd.DataFrame([test_metrics]).to_csv(SUMMARY_PATH, index=False)
    test_report.to_csv(REPORT_PATH)
    test_cm.to_csv(CONFUSION_PATH)

    log_step(f"Saved Hugging Face model directory to: {MODEL_DIR}")
    log_step(f"Saved model bundle to: {MODEL_PATH}")
    log_step(f"Saved training log to: {SEARCH_LOG_PATH}")
    log_step(f"Saved summary to: {SUMMARY_PATH}")
    log_step(f"Saved classification report to: {REPORT_PATH}")
    log_step(f"Saved confusion matrix to: {CONFUSION_PATH}")

    log_step("Step 12/12: Done")
    print(f"[{now_str()}] Best epoch: {best_epoch}", flush=True)
    print(
        f"[{now_str()}] Total runtime: {format_seconds(time.time() - overall_start)}",
        flush=True,
    )


if __name__ == "__main__":
    main()