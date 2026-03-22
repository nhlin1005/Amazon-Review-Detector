
import json
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, OneHotEncoder
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

DATA_DIR = Path("dataset")
TRAIN_PATH = DATA_DIR / "amazon_reviews_train.csv"
VAL_PATH = DATA_DIR / "amazon_reviews_val.csv"
TEST_PATH = DATA_DIR / "amazon_reviews_test.csv"

MODEL_PATH = DATA_DIR / "best_single_review_model_v3.joblib"
SUMMARY_PATH = DATA_DIR / "best_single_model_v3_results.csv"
REPORT_PATH = DATA_DIR / "best_single_model_v3_classification_report.csv"
CONFUSION_PATH = DATA_DIR / "best_single_model_v3_confusion_matrix.csv"
SEARCH_LOG_PATH = DATA_DIR / "best_single_model_v3_search_log.csv"


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


def now_str():
    return datetime.now().strftime("%H:%M:%S")


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


def build_class_weight(label_encoder, neutral_weight=1.0, negative_weight=1.0, positive_weight=1.0):
    class_to_index = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
    weights_by_name = {
        "negative": negative_weight,
        "neutral": neutral_weight,
        "positive": positive_weight,
    }
    return {class_to_index[name]: weight for name, weight in weights_by_name.items() if name in class_to_index}


def build_pipeline(config, label_encoder):
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

    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=1.0)

    class_weight = build_class_weight(
        label_encoder,
        neutral_weight=config["neutral_weight"],
        negative_weight=config["negative_weight"],
        positive_weight=config["positive_weight"],
    )

    model = LinearSVC(
        C=config["c_value"],
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        max_iter=config["max_iter"],
    )

    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("clf", model),
        ]
    )


def make_configs():
    configs = [
        {"word_max_features": 30000, "char_max_features": 20000, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5),
         "word_min_df": 3, "char_min_df": 3, "c_value": 0.30, "neutral_weight": 1.20, "negative_weight": 1.00,
         "positive_weight": 0.90, "use_text_stats": True, "max_iter": 6000},
        {"word_max_features": 30000, "char_max_features": 20000, "word_ngram_range": (1, 2), "char_ngram_range": (3, 5),
         "word_min_df": 3, "char_min_df": 3, "c_value": 0.40, "neutral_weight": 1.30, "negative_weight": 1.00,
         "positive_weight": 0.90, "use_text_stats": True, "max_iter": 7000},
        {"word_max_features": 50000, "char_max_features": 30000, "word_ngram_range": (1, 3), "char_ngram_range": (3, 5),
         "word_min_df": 3, "char_min_df": 3, "c_value": 0.50, "neutral_weight": 1.25, "negative_weight": 1.00,
         "positive_weight": 0.90, "use_text_stats": True, "max_iter": 7000},
        {"word_max_features": 50000, "char_max_features": 30000, "word_ngram_range": (1, 3), "char_ngram_range": (3, 6),
         "word_min_df": 2, "char_min_df": 3, "c_value": 0.45, "neutral_weight": 1.35, "negative_weight": 1.05,
         "positive_weight": 0.88, "use_text_stats": True, "max_iter": 8000},
        {"word_max_features": 50000, "char_max_features": 50000, "word_ngram_range": (1, 2), "char_ngram_range": (2, 5),
         "word_min_df": 2, "char_min_df": 2, "c_value": 0.45, "neutral_weight": 1.30, "negative_weight": 1.05,
         "positive_weight": 0.90, "use_text_stats": True, "max_iter": 8000},
        {"word_max_features": 70000, "char_max_features": 30000, "word_ngram_range": (1, 3), "char_ngram_range": (3, 5),
         "word_min_df": 2, "char_min_df": 3, "c_value": 0.55, "neutral_weight": 1.15, "negative_weight": 1.10,
         "positive_weight": 0.92, "use_text_stats": True, "max_iter": 8000},
        {"word_max_features": 70000, "char_max_features": 40000, "word_ngram_range": (1, 3), "char_ngram_range": (3, 6),
         "word_min_df": 2, "char_min_df": 3, "c_value": 0.60, "neutral_weight": 1.40, "negative_weight": 1.05,
         "positive_weight": 0.88, "use_text_stats": True, "max_iter": 9000},
        {"word_max_features": 80000, "char_max_features": 40000, "word_ngram_range": (1, 3), "char_ngram_range": (3, 6),
         "word_min_df": 2, "char_min_df": 3, "c_value": 0.65, "neutral_weight": 1.35, "negative_weight": 1.05,
         "positive_weight": 0.88, "use_text_stats": True, "max_iter": 9000},
        {"word_max_features": 90000, "char_max_features": 50000, "word_ngram_range": (1, 3), "char_ngram_range": (3, 6),
         "word_min_df": 2, "char_min_df": 2, "c_value": 0.50, "neutral_weight": 1.45, "negative_weight": 1.10,
         "positive_weight": 0.85, "use_text_stats": True, "max_iter": 10000},
        {"word_max_features": 100000, "char_max_features": 60000, "word_ngram_range": (1, 3), "char_ngram_range": (2, 6),
         "word_min_df": 2, "char_min_df": 2, "c_value": 0.55, "neutral_weight": 1.50, "negative_weight": 1.10,
         "positive_weight": 0.85, "use_text_stats": True, "max_iter": 10000},
        {"word_max_features": 70000, "char_max_features": 50000, "word_ngram_range": (1, 2), "char_ngram_range": (2, 5),
         "word_min_df": 2, "char_min_df": 2, "c_value": 0.35, "neutral_weight": 1.55, "negative_weight": 1.05,
         "positive_weight": 0.88, "use_text_stats": True, "max_iter": 9000},
        {"word_max_features": 120000, "char_max_features": 70000, "word_ngram_range": (1, 3), "char_ngram_range": (2, 6),
         "word_min_df": 2, "char_min_df": 2, "c_value": 0.40, "neutral_weight": 1.60, "negative_weight": 1.10,
         "positive_weight": 0.85, "use_text_stats": True, "max_iter": 12000},
    ]
    return configs


def main():
    overall_start = time.time()

    print(f"[{now_str()}] Loading train/val/test CSV files...")
    train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
    val_df = pd.read_csv(VAL_PATH, low_memory=False)
    test_df = pd.read_csv(TEST_PATH, low_memory=False)

    text_col = choose_text_column(train_df)
    if text_col != "clean_text":
        print(f"[{now_str()}] clean_text not found. Copying from '{text_col}'...")
        train_df["clean_text"] = train_df[text_col].fillna("").astype(str)
        val_df["clean_text"] = val_df[text_col].fillna("").astype(str)
        test_df["clean_text"] = test_df[text_col].fillna("").astype(str)

    for df in (train_df, val_df, test_df):
        df["clean_text"] = df["clean_text"].fillna("").astype(str)
        df["category"] = df["category"].fillna("unknown").astype(str)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_val = label_encoder.transform(val_df["sentiment"])
    y_test = label_encoder.transform(test_df["sentiment"])
    label_names = list(label_encoder.classes_)

    X_train = train_df[["clean_text", "category"]]
    X_val = val_df[["clean_text", "category"]]
    X_test = test_df[["clean_text", "category"]]

    configs = make_configs()

    print(f"[{now_str()}] Using text column: clean_text")
    print(f"[{now_str()}] Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")
    print(f"[{now_str()}] Label order: {label_names}")
    print(f"[{now_str()}] Starting extended search with {len(configs)} configs...")

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
        print(f"[{now_str()}] Training config {i}/{len(configs)}")
        print(json.dumps(config))
        print(f"[{now_str()}] Building pipeline...")

        model = build_pipeline(config, label_encoder)

        print(f"[{now_str()}] Fitting model...")
        model.fit(X_train, y_train)

        fit_elapsed = time.time() - config_start
        print(f"[{now_str()}] Fit finished in {fit_elapsed:.1f}s. Running validation...")

        y_val_pred = model.predict(X_val)
        val_metrics, val_report, val_cm = compute_metrics(y_val, y_val_pred, label_names)
        neutral_f1 = float(val_report.loc["neutral", "f1-score"]) if "neutral" in val_report.index else np.nan
        negative_f1 = float(val_report.loc["negative", "f1-score"]) if "negative" in val_report.index else np.nan
        positive_f1 = float(val_report.loc["positive", "f1-score"]) if "positive" in val_report.index else np.nan

        print_metrics(f"Validation | config {i}/{len(configs)}", val_metrics, val_report, val_cm)
        print(f"[{now_str()}] Per-class F1 -> negative: {negative_f1:.4f}, neutral: {neutral_f1:.4f}, positive: {positive_f1:.4f}")

        row = dict(config)
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["val_negative_f1"] = negative_f1
        row["val_neutral_f1"] = neutral_f1
        row["val_positive_f1"] = positive_f1
        row["elapsed_seconds"] = round(time.time() - config_start, 2)
        search_rows.append(row)

        if val_metrics["macro_f1"] > best_val_score:
            best_val_score = val_metrics["macro_f1"]
            best_model = model
            best_config = config
            best_val_metrics = val_metrics
            best_val_report = val_report
            best_val_cm = val_cm
            print(f"[{now_str()}] New best model found. Best validation macro_f1 = {best_val_score:.4f}")
        else:
            print(f"[{now_str()}] Best validation macro_f1 still = {best_val_score:.4f}")

        completed = i
        avg_time = (time.time() - overall_start) / completed
        remaining = avg_time * (len(configs) - completed)
        print(f"[{now_str()}] Progress: {completed}/{len(configs)} configs complete. Avg/config = {avg_time:.1f}s. Estimated remaining = {remaining:.1f}s.")

    print("\n" + "#" * 110)
    print(f"[{now_str()}] Best validation config:")
    print(json.dumps(best_config))
    print(f"[{now_str()}] Best validation macro_f1: {best_val_score:.4f}")
    print(f"[{now_str()}] Running final test evaluation...")

    y_test_pred = best_model.predict(X_test)
    test_metrics, test_report, test_cm = compute_metrics(y_test, y_test_pred, label_names)
    print_metrics("Test | best config", test_metrics, test_report, test_cm)

    bundle = {
        "model": best_model,
        "label_encoder": label_encoder,
        "label_names": label_names,
        "best_config": best_config,
        "best_val_metrics": best_val_metrics,
        "best_val_report": best_val_report.to_dict(),
        "text_column": "clean_text",
        "feature_columns": ["clean_text", "category"],
    }
    joblib.dump(bundle, MODEL_PATH)

    summary_df = pd.DataFrame(
        [
            {"split": "validation", **best_val_metrics, "config_json": json.dumps(best_config)},
            {"split": "test", **test_metrics, "config_json": json.dumps(best_config)},
        ]
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)
    test_report.reset_index().rename(columns={"index": "label"}).to_csv(REPORT_PATH, index=False)
    test_cm.to_csv(CONFUSION_PATH, index=True)
    pd.DataFrame(search_rows).to_csv(SEARCH_LOG_PATH, index=False)

    total_elapsed = time.time() - overall_start
    print("\n" + "#" * 110)
    print(f"[{now_str()}] Saved model bundle to: {MODEL_PATH}")
    print(f"[{now_str()}] Saved summary to: {SUMMARY_PATH}")
    print(f"[{now_str()}] Saved classification report to: {REPORT_PATH}")
    print(f"[{now_str()}] Saved confusion matrix to: {CONFUSION_PATH}")
    print(f"[{now_str()}] Saved search log to: {SEARCH_LOG_PATH}")
    print(f"[{now_str()}] Total elapsed time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
