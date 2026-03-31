import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder

# Optional RAM monitor
try:
    import psutil
except ImportError:
    psutil = None

RANDOM_STATE = 42

DATA_DIR = Path("dataset")
TRAIN_PATH = DATA_DIR / "amazon_reviews_train.csv"
VAL_PATH = DATA_DIR / "amazon_reviews_val.csv"
TEST_PATH = DATA_DIR / "amazon_reviews_test.csv"

MODEL_PATH = DATA_DIR / "best_single_review_model_nn.joblib"
SUMMARY_PATH = DATA_DIR / "best_single_model_nn_results.csv"
REPORT_PATH = DATA_DIR / "best_single_model_nn_classification_report.csv"
CONFUSION_PATH = DATA_DIR / "best_single_model_nn_confusion_matrix.csv"
SEARCH_LOG_PATH = DATA_DIR / "best_single_model_nn_search_log.csv"


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


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = pd.Series(X).fillna("").astype(str)

        char_len = series.str.len().astype(float)
        word_count = series.str.split().str.len().astype(float)
        exclam_count = series.str.count("!").astype(float)
        question_count = series.str.count(r"\?").astype(float)

        token_lists = series.str.split()
        unique_ratio = []
        avg_word_len = []
        digit_ratio = []
        uppercase_ratio = []

        for text, tokens in zip(series.tolist(), token_lists.tolist()):
            if tokens:
                unique_ratio.append(len(set(tokens)) / len(tokens))
                avg_word_len.append(sum(len(t) for t in tokens) / len(tokens))
            else:
                unique_ratio.append(0.0)
                avg_word_len.append(0.0)

            if text:
                digit_ratio.append(sum(ch.isdigit() for ch in text) / len(text))
                letters = [ch for ch in text if ch.isalpha()]
                if letters:
                    uppercase_ratio.append(sum(ch.isupper() for ch in letters) / len(letters))
                else:
                    uppercase_ratio.append(0.0)
            else:
                digit_ratio.append(0.0)
                uppercase_ratio.append(0.0)

        arr = np.column_stack(
            [
                char_len.to_numpy(),
                word_count.to_numpy(),
                exclam_count.to_numpy(),
                question_count.to_numpy(),
                np.asarray(unique_ratio, dtype=float),
                np.asarray(avg_word_len, dtype=float),
                np.asarray(digit_ratio, dtype=float),
                np.asarray(uppercase_ratio, dtype=float),
            ]
        )
        return sparse.csr_matrix(arr)


def choose_text_column(df):
    if "clean_text" in df.columns:
        return "clean_text"
    if "text" in df.columns:
        return "text"
    raise ValueError("Dataset must contain either 'clean_text' or 'text'.")


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


def build_pipeline(config):
    transformers = [
        (
            "word_tfidf",
            TfidfVectorizer(
                lowercase=False,
                analyzer="word",
                ngram_range=config["word_ngram_range"],
                min_df=config["word_min_df"],
                max_df=0.98,
                max_features=config["word_max_features"],
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            "clean_text",
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                lowercase=False,
                analyzer="char_wb",
                ngram_range=config["char_ngram_range"],
                min_df=config["char_min_df"],
                max_df=1.0,
                max_features=config["char_max_features"],
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            "clean_text",
        ),
        (
            "category_ohe",
            make_one_hot_encoder(),
            ["category"],
        ),
    ]

    if config["use_text_stats"]:
        transformers.append(
            (
                "text_stats",
                Pipeline(
                    steps=[
                        ("stats", TextStatsTransformer()),
                        ("scale", MaxAbsScaler()),
                    ]
                ),
                "clean_text",
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        sparse_threshold=1.0,
    )

    model = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation=config["activation"],
        solver="adam",
        alpha=config["alpha"],
        batch_size=config["batch_size"],
        learning_rate_init=config["learning_rate_init"],
        max_iter=config["max_iter"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=RANDOM_STATE,
        verbose=True,
    )

    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("scale_all", MaxAbsScaler()),
            ("clf", model),
        ]
    )


def make_configs():
    return [
        {
            "word_max_features": 30000,
            "char_max_features": 15000,
            "word_ngram_range": (1, 2),
            "char_ngram_range": (3, 5),
            "word_min_df": 3,
            "char_min_df": 3,
            "use_text_stats": True,
            "hidden_layer_sizes": (256,),
            "activation": "relu",
            "alpha": 1e-4,
            "batch_size": 256,
            "learning_rate_init": 1e-3,
            "max_iter": 20,
        },
        {
            "word_max_features": 40000,
            "char_max_features": 20000,
            "word_ngram_range": (1, 2),
            "char_ngram_range": (3, 5),
            "word_min_df": 3,
            "char_min_df": 3,
            "use_text_stats": True,
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "alpha": 3e-4,
            "batch_size": 256,
            "learning_rate_init": 8e-4,
            "max_iter": 20,
        },
        {
            "word_max_features": 50000,
            "char_max_features": 25000,
            "word_ngram_range": (1, 3),
            "char_ngram_range": (3, 5),
            "word_min_df": 2,
            "char_min_df": 3,
            "use_text_stats": True,
            "hidden_layer_sizes": (512, 128),
            "activation": "relu",
            "alpha": 5e-4,
            "batch_size": 256,
            "learning_rate_init": 6e-4,
            "max_iter": 25,
        },
    ]


def main():
    overall_start = time.time()

    log_step("Step 1/9: Loading train/val/test CSV files")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VAL_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    log_step("Step 2/9: Checking text column")
    text_col = choose_text_column(train_df)
    if text_col != "clean_text":
        log_step(f"clean_text not found, copying from '{text_col}'")
        train_df["clean_text"] = train_df[text_col].fillna("").astype(str)
        val_df["clean_text"] = val_df[text_col].fillna("").astype(str)
        test_df["clean_text"] = test_df[text_col].fillna("").astype(str)

    log_step("Step 3/9: Cleaning text and category columns")
    for df in (train_df, val_df, test_df):
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
        df["category"] = df["category"].fillna("unknown").astype(str)

    log_step("Step 4/9: Encoding labels")
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_val = label_encoder.transform(val_df["sentiment"])
    y_test = label_encoder.transform(test_df["sentiment"])
    label_names = list(label_encoder.classes_)

    X_train = train_df[["clean_text", "category"]]
    X_val = val_df[["clean_text", "category"]]
    X_test = test_df[["clean_text", "category"]]

    configs = make_configs()

    print()
    print(f"[{now_str()}] Using text column: clean_text")
    print(f"[{now_str()}] Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")
    print(f"[{now_str()}] Label order: {label_names}")
    print(f"[{now_str()}] Starting NN search with {len(configs)} configs...", flush=True)

    best_model = None
    best_config = None
    best_val_metrics = None
    best_val_report = None
    best_val_cm = None
    best_val_score = -1.0
    search_rows = []

    for i, config in enumerate(configs, start=1):
        config_start = time.time()

        print("\n" + "=" * 110)
        log_step(f"Config {i}/{len(configs)}: Building pipeline")
        print(json.dumps(config), flush=True)

        model = build_pipeline(config)

        log_step(f"Config {i}/{len(configs)}: Fitting model")
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_elapsed = time.time() - fit_start
        log_step(f"Config {i}/{len(configs)}: Fit finished in {format_seconds(fit_elapsed)}")

        log_step(f"Config {i}/{len(configs)}: Predicting validation set")
        pred_start = time.time()
        y_val_pred = model.predict(X_val)
        pred_elapsed = time.time() - pred_start
        log_step(f"Config {i}/{len(configs)}: Validation prediction finished in {format_seconds(pred_elapsed)}")

        log_step(f"Config {i}/{len(configs)}: Computing validation metrics")
        val_metrics, val_report, val_cm = compute_metrics(y_val, y_val_pred, label_names)

        row = dict(config)
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["fit_seconds"] = round(fit_elapsed, 2)
        row["predict_seconds"] = round(pred_elapsed, 2)
        row["elapsed_seconds"] = round(time.time() - config_start, 2)
        search_rows.append(row)

        print_metrics(f"Validation | config {i}/{len(configs)}", val_metrics, val_report, val_cm)

        if val_metrics["macro_f1"] > best_val_score:
            best_val_score = val_metrics["macro_f1"]
            best_model = model
            best_config = config
            best_val_metrics = val_metrics
            best_val_report = val_report
            best_val_cm = val_cm
            log_step(
                f"Config {i}/{len(configs)}: New best model found "
                f"(validation macro_f1 = {best_val_score:.4f})"
            )
        else:
            log_step(
                f"Config {i}/{len(configs)}: Not best, current best validation macro_f1 = {best_val_score:.4f}"
            )

        log_step(f"Config {i}/{len(configs)}: Total elapsed {format_seconds(time.time() - config_start)}")

    print("\n" + "=" * 110)
    log_step("Step 5/9: Best config selected")
    print(json.dumps(best_config, indent=2), flush=True)

    log_step("Step 6/9: Running final test evaluation")
    test_pred_start = time.time()
    y_test_pred = best_model.predict(X_test)
    log_step(f"Test prediction finished in {format_seconds(time.time() - test_pred_start)}")

    log_step("Step 7/9: Computing test metrics")
    test_metrics, test_report, test_cm = compute_metrics(y_test, y_test_pred, label_names)
    print_metrics("Final Test Results", test_metrics, test_report, test_cm)

    bundle = {
        "model": best_model,
        "label_encoder": label_encoder,
        "label_names": label_names,
        "best_config": best_config,
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    log_step("Step 8/9: Saving model and reports")
    joblib.dump(bundle, MODEL_PATH)
    pd.DataFrame(search_rows).to_csv(SEARCH_LOG_PATH, index=False)
    pd.DataFrame([test_metrics]).to_csv(SUMMARY_PATH, index=False)
    test_report.to_csv(REPORT_PATH)
    test_cm.to_csv(CONFUSION_PATH)

    log_step(f"Saved model bundle to: {MODEL_PATH}")
    log_step(f"Saved search log to: {SEARCH_LOG_PATH}")
    log_step(f"Saved summary to: {SUMMARY_PATH}")
    log_step(f"Saved classification report to: {REPORT_PATH}")
    log_step(f"Saved confusion matrix to: {CONFUSION_PATH}")

    log_step("Step 9/9: Done")
    print(f"[{now_str()}] Total runtime: {format_seconds(time.time() - overall_start)}", flush=True)


if __name__ == "__main__":
    main()