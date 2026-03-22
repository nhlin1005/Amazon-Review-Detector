import json
import re
from pathlib import Path

import pandas as pd

RAW_JSON_PATH = r"dataset\Arts_Crafts_and_Sewing_5.json\Arts_Crafts_and_Sewing_5.json"
OUTPUT_CSV_PATH = r"dataset\new_amazon_test_ready.csv"
DATASET_DISPLAY_NAME = None

MAX_ROWS = 50000
RANDOM_STATE = 42


def infer_display_name(raw_path_str):
    path = Path(raw_path_str)
    stem = path.stem
    parent_name = path.parent.name

    if parent_name and parent_name != path.name:
        return parent_name.replace("_5", "").replace("_", " ")
    return stem.replace("_5", "").replace("_", " ")


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_sentiment(rating):
    try:
        rating = float(rating)
    except Exception:
        return None

    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def main():
    raw_path = Path(RAW_JSON_PATH)
    output_path = Path(OUTPUT_CSV_PATH)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw JSON file not found: {raw_path}")

    dataset_name = DATASET_DISPLAY_NAME or infer_display_name(RAW_JSON_PATH)

    print(f"Reading raw JSON from: {raw_path}")
    print(f"Dataset display name: {dataset_name}")

    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)

            summary = str(obj.get("summary", "") or "").strip()
            review_text = str(obj.get("reviewText", "") or "").strip()

            if summary and review_text:
                text = f"{summary}. {review_text}"
            elif summary:
                text = summary
            else:
                text = review_text

            row = {
                "category": dataset_name,
                "asin": obj.get("asin", ""),
                "overall": obj.get("overall", None),
                "sentiment": rating_to_sentiment(obj.get("overall", None)),
                "verified": obj.get("verified", None),
                "vote": obj.get("vote", None),
                "unixReviewTime": obj.get("unixReviewTime", None),
                "text": text,
                "clean_text": clean_text(text),
            }
            rows.append(row)

            if i % 500000 == 0:
                print(f"Loaded {i:,} raw rows...")

    df = pd.DataFrame(rows)
    print("\nBefore dropping empty clean_text:", df.shape)

    df = df[df["clean_text"].str.len() > 0].copy()
    print("After dropping empty clean_text:", df.shape)

    if MAX_ROWS is not None and len(df) > MAX_ROWS:
        print(f"Sampling down to {MAX_ROWS:,} rows...")
        df = df.sample(n=MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\nFinal shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts(dropna=False))

    print("\nFirst 3 rows:")
    print(df.head(3)[["category", "overall", "sentiment", "text", "clean_text"]])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved processed CSV to: {output_path}")


if __name__ == "__main__":
    main()