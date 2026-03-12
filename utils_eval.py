import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


def evaluate_model(model_name, y_true, y_pred, label_names):
    accuracy = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print()
    print(model_name, ":  ")
    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(p_macro, 4))
    print("Macro Recall:", round(r_macro, 4))
    print("Macro F1:", round(f1_macro, 4))
    print("Weighted F1:", round(f1_weighted, 4))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

    print("\nConfusion Matrix:")
    print(cm_df)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted
    }