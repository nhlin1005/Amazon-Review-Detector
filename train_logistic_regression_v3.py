from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from utils_eval_v2 import evaluate_model

DATA_DIR = Path("dataset")
TEXT_COLUMN = "clean_text"
RANDOM_STATE = 42


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(DATA_DIR / "amazon_reviews_train.csv", low_memory=False)
    val_df = pd.read_csv(DATA_DIR / "amazon_reviews_val.csv", low_memory=False)
    test_df = pd.read_csv(DATA_DIR / "amazon_reviews_test.csv", low_memory=False)
    return train_df, val_df, test_df


def build_pipeline(max_features: int, min_df: int, c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=c_value,
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        solver="liblinear",
                    )
                ),
            ),
        ]
    )


def main() -> None:
    train_df, val_df, test_df = load_splits()

    X_train = train_df[TEXT_COLUMN].fillna("")
    X_val = val_df[TEXT_COLUMN].fillna("")
    X_test = test_df[TEXT_COLUMN].fillna("")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_val = label_encoder.transform(val_df["sentiment"])
    y_test = label_encoder.transform(test_df["sentiment"])
    label_names = list(label_encoder.classes_)

    configs = [
        {"max_features": 10000, "min_df": 5, "c_value": 1.0},
        {"max_features": 15000, "min_df": 5, "c_value": 2.0},
        {"max_features": 20000, "min_df": 3, "c_value": 2.0},
    ]

    best_model = None
    best_result = None
    best_config = None

    print(f"Using text column: {TEXT_COLUMN}")
    print(f"Train/Val/Test sizes: {len(train_df)}, {len(val_df)}, {len(test_df)}")

    for i, config in enumerate(configs, start=1):
        print(f"\nTraining Logistic Regression config {i}/{len(configs)}: {config}")
        model = build_pipeline(**config)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        result = evaluate_model(
            model_name=f"TF-IDF + Logistic Regression (Val) | {config}",
            y_true=y_val,
            y_pred=val_pred,
            label_names=label_names,
        )

        if best_result is None or result["macro_f1"] > best_result["macro_f1"]:
            best_model = model
            best_result = result
            best_config = config

    print(f"\nBest validation config: {best_config}")
    print(f"Best validation macro F1: {best_result['macro_f1']:.4f}")

    test_pred = best_model.predict(X_test)
    test_result = evaluate_model(
        model_name=f"TF-IDF + Logistic Regression (Test) | best={best_config}",
        y_true=y_test,
        y_pred=test_pred,
        label_names=label_names,
        output_dir=DATA_DIR,
        output_prefix="logistic_best",
    )
    test_result["text_column"] = TEXT_COLUMN
    test_result["best_config"] = str(best_config)
    pd.DataFrame([test_result]).to_csv(DATA_DIR / "simple_logistic_results_v2.csv", index=False)
    print("\nSaved summary to dataset/simple_logistic_results_v2.csv")


if __name__ == "__main__":
    main()
