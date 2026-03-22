
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = r"dataset\combined_review_credibility_model.joblib"
INPUT_CSV_PATH = r"dataset\new_amazon_test_ready.csv"
OUTPUT_FLAGGED_CSV_PATH = r"dataset\new_amazon_test_with_credibility.csv"
OUTPUT_SUMMARY_CSV_PATH = r"dataset\new_amazon_test_credibility_summary.csv"
OUTPUT_EXAMPLES_DIR = r"dataset\credibility_examples"
NUM_EXAMPLES_PER_GROUP = 10


def main():
    model_path = Path(MODEL_PATH)
    input_path = Path(INPUT_CSV_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    bundle = joblib.load(model_path)
    scorer = bundle["scorer"]

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print("Input shape:", df.shape)

    print("Scoring low credibility...")
    scored = scorer.predict(df)

    flagged_path = Path(OUTPUT_FLAGGED_CSV_PATH)
    flagged_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(flagged_path, index=False, encoding="utf-8")
    print(f"Saved scored dataset to: {flagged_path}")

    summary_rows = []

    overall = {
        "group_name": "ALL",
        "row_count": int(len(scored)),
        "suspicious_rate": round((scored["suspicious_label"] == "suspicious").mean(), 5),
        "high_risk_rate": round((scored["low_credibility_label"] == "high").mean(), 5),
        "avg_score": round(scored["low_credibility_score"].mean(), 5),
    }
    summary_rows.append(overall)

    if "category" in scored.columns:
        for category, g in scored.groupby("category"):
            summary_rows.append(
                {
                    "group_name": str(category),
                    "row_count": int(len(g)),
                    "suspicious_rate": round((g["suspicious_label"] == "suspicious").mean(), 5),
                    "high_risk_rate": round((g["low_credibility_label"] == "high").mean(), 5),
                    "avg_score": round(g["low_credibility_score"].mean(), 5),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(OUTPUT_SUMMARY_CSV_PATH)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    print(summary.to_string(index=False))

    examples_dir = Path(OUTPUT_EXAMPLES_DIR)
    examples_dir.mkdir(parents=True, exist_ok=True)

    suspicious = scored[scored["suspicious_label"] == "suspicious"].copy()
    not_suspicious = scored[scored["suspicious_label"] == "not_suspicious"].copy()

    suspicious = suspicious.sort_values(
        by=["low_credibility_score", "word_count"], ascending=[False, True]
    ).head(NUM_EXAMPLES_PER_GROUP)
    not_suspicious = not_suspicious.sample(
        n=min(NUM_EXAMPLES_PER_GROUP, len(not_suspicious)),
        random_state=42,
    ) if len(not_suspicious) > 0 else not_suspicious

    suspicious.to_csv(examples_dir / "suspicious_examples.csv", index=False, encoding="utf-8")
    not_suspicious.to_csv(examples_dir / "not_suspicious_examples.csv", index=False, encoding="utf-8")
    print(f"Saved example CSVs to: {examples_dir}")

    print("\nTop suspicious examples:")
    cols = [c for c in ["category", "text", "predicted_sentiment", "low_credibility_score", "low_credibility_label", "suspicious_reasons"] if c in suspicious.columns]
    if len(suspicious) > 0:
        print(suspicious[cols].head(5).to_string(index=False))
    else:
        print("No suspicious rows found.")


if __name__ == "__main__":
    main()
