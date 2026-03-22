from __future__ import annotations

from pathlib import Path
import json
import re

import pandas as pd

DATA_DIR = Path("dataset")
SOURCE_FILES = {
    "Cell Phones and Accessories": DATA_DIR / "Cell_Phones_and_Accessories_5.json" / "Cell_Phones_and_Accessories_5.json",
    "Industrial and Scientific": DATA_DIR / "Industrial_and_Scientific_5.json" / "Industrial_and_Scientific_5.json",
    "Sports and Outdoors": DATA_DIR / "Sports_and_Outdoors_5.json" / "Sports_and_Outdoors_5.json",
}
OUTPUT_PATH = DATA_DIR / "amazon_reviews_cleaned.csv"
NEEDED_COLS = ["asin", "reviewText", "summary", "overall", "verified", "vote", "unixReviewTime"]


RE_MULTISPACE = re.compile(r"\s+")
RE_NONWORD = re.compile(r"[^\w\s]")


def rating_to_sentiment(rating):
    if rating in [1.0, 2.0, 1, 2]:
        return "negative"
    if rating in [3.0, 3]:
        return "neutral"
    if rating in [4.0, 5.0, 4, 5]:
        return "positive"
    return None


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = RE_MULTISPACE.sub(" ", text)
    text = RE_NONWORD.sub("", text)
    return text.strip()



def load_amazon_reviews(file_path: Path, category_name: str) -> pd.DataFrame:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            row = {col: review.get(col) for col in NEEDED_COLS}
            row["category"] = category_name
            rows.append(row)
    return pd.DataFrame(rows)



def main() -> None:
    dfs = []
    for category_name, file_path in SOURCE_FILES.items():
        df = load_amazon_reviews(file_path, category_name)
        print(f"{category_name} shape: {df.shape}")

        df["summary"] = df["summary"].fillna("")
        df["reviewText"] = df["reviewText"].fillna("")
        df["text"] = (df["summary"] + " " + df["reviewText"]).str.strip()
        df["clean_text"] = df["text"].apply(clean_text)
        df["sentiment"] = df["overall"].apply(rating_to_sentiment)
        dfs.append(df)

    all_reviews_df = pd.concat(dfs, ignore_index=True)

    print("\nCombined dataset shape:", all_reviews_df.shape)
    print("\nCombined category counts:")
    print(all_reviews_df["category"].value_counts())
    print("\nCombined sentiment counts:")
    print(all_reviews_df["sentiment"].value_counts())
    print("\nExample rows:")
    print(all_reviews_df[["category", "overall", "sentiment", "text", "clean_text"]].head())

    all_reviews_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
