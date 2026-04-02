from pathlib import Path
import time
import __main__

import pandas as pd

from credibility_model import CredibilityConfig, ReviewCredibilityScorer
from train_roberta_model import RobertaSentimentPredictor

__main__.RobertaSentimentPredictor = RobertaSentimentPredictor

INPUT_PATH = r"dataset\amazon_reviews_ready.csv"
SENTIMENT_BUNDLE_PATH = r"dataset\best_single_review_model_roberta_fast.joblib"
OUTPUT_PATH = r"dataset\amazon_reviews_flagged_v2.csv"
SUMMARY_PATH = r"dataset\suspicious_review_summary_v2.csv"

# Change this if you want bigger/smaller chunks
CHUNK_SIZE = 200


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds % 60
    return f"{minutes}m {remain:.1f}s"


def main():
    overall_start = time.time()

    input_path = Path(INPUT_PATH)
    sentiment_path = Path(SENTIMENT_BUNDLE_PATH)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment bundle not found: {sentiment_path}")

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df)} rows")

    print(f"Loading Transformer sentiment bundle from: {sentiment_path}")
    print("Building combined scorer...")
    scorer = ReviewCredibilityScorer.from_sentiment_bundle_path(
        str(sentiment_path),
        config=CredibilityConfig(),
    )

    print("Scoring dataset in chunks...")
    scored_parts = []
    total_rows = len(df)
    total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE

    scoring_start = time.time()

    for chunk_idx, start_idx in enumerate(range(0, total_rows, CHUNK_SIZE), start=1):
        end_idx = min(start_idx + CHUNK_SIZE, total_rows)
        chunk = df.iloc[start_idx:end_idx].copy()

        chunk_start = time.time()
        print(
            f"[Chunk {chunk_idx}/{total_chunks}] "
            f"Scoring rows {start_idx} to {end_idx - 1} "
            f"({len(chunk)} rows)..."
        )

        scored_chunk = scorer.predict(chunk)
        scored_parts.append(scored_chunk)

        elapsed = time.time() - chunk_start
        total_elapsed = time.time() - scoring_start
        done_rows = end_idx
        pct = 100.0 * done_rows / total_rows

        print(
            f"[Chunk {chunk_idx}/{total_chunks}] Done in {format_seconds(elapsed)} | "
            f"Progress: {done_rows}/{total_rows} rows ({pct:.1f}%) | "
            f"Total scoring time: {format_seconds(total_elapsed)}"
        )

    scored = pd.concat(scored_parts, ignore_index=True)
    print(f"Finished scoring all rows in {format_seconds(time.time() - scoring_start)}")

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

    if "category" in scored.columns:
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

    print(f"\nTotal runtime: {format_seconds(time.time() - overall_start)}")


if __name__ == "__main__":
    main()
