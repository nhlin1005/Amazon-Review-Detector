from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(
    model_name: str,
    y_true,
    y_pred,
    label_names,
    output_dir: str | Path | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

    print(f"\n{model_name}")
    print("-" * len(model_name))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {p_macro:.4f}")
    print(f"Macro Recall: {r_macro:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted Precision: {p_weighted:.4f}")
    print(f"Weighted Recall: {r_weighted:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(report_df.round(4))

    print("\nConfusion Matrix:")
    print(cm_df)

    result = {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_precision": p_weighted,
        "weighted_recall": r_weighted,
        "weighted_f1": f1_weighted,
    }

    if output_dir is not None and output_prefix is not None:
        pd.DataFrame([result]).to_csv(output_dir / f"{output_prefix}_summary.csv", index=False)
        report_df.to_csv(output_dir / f"{output_prefix}_classification_report.csv")
        cm_df.to_csv(output_dir / f"{output_prefix}_confusion_matrix.csv")

    return result
