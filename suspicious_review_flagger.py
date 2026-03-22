
from pathlib import Path

import pandas as pd

from credibility_model import CredibilityConfig, ReviewCredibilityScorer

INPUT_PATH = r"dataset\amazon_reviews_ready.csv"
SENTIMENT_BUNDLE_PATH = r"dataset\best_single_review_model_v3.joblib"
OUTPUT_PATH = r"dataset\amazon_reviews_flagged_v2.csv"
SUMMARY_PATH = r"dataset\suspicious_review_summary_v2.csv"


def main():
    input_path = Path(INPUT_PATH)
    sentiment_path = Path(SENTIMENT_BUNDLE_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment bundle not found: {sentiment_path}")

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    print("Loading sentiment bundle and building combined scorer...")
    scorer = ReviewCredibilityScorer.from_sentiment_bundle_path(
        str(sentiment_path),
        config=CredibilityConfig(),
    )

    print("Scoring dataset...")
    scored = scorer.predict(df)

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved flagged dataset to: {output_path}")

    rows = [
        {
            "group_name": "ALL",
            "row_count": int(len(scored)),
            "suspicious_rate": round((scored["suspicious_label"] == "suspicious").mean(), 5),
            "high_risk_rate": round((scored["low_credibility_label"] == "high").mean(), 5),
            "avg_score": round(scored["low_credibility_score"].mean(), 5),
        }
    ]
    for category, g in scored.groupby("category"):
        rows.append(
            {
                "group_name": str(category),
                "row_count": int(len(g)),
                "suspicious_rate": round((g["suspicious_label"] == "suspicious").mean(), 5),
                "high_risk_rate": round((g["low_credibility_label"] == "high").mean(), 5),
                "avg_score": round(g["low_credibility_score"].mean(), 5),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = Path(SUMMARY_PATH)
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved summary to: {summary_path}")

    print("\nOverall suspicious rate:", round((scored["suspicious_label"] == "suspicious").mean(), 4))
    print("\nRisk label distribution:")
    print(scored["low_credibility_label"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
