from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("dataset")
SAMPLE_SIZE_PER_CATEGORY = 40000
FINAL_COLUMNS = [
    "category",
    "asin",
    "overall",
    "sentiment",
    "verified",
    "vote",
    "unixReviewTime",
    "text",
    "clean_text",
]


def sample_one_category(df: pd.DataFrame, category_name: str, n: int) -> pd.DataFrame:
    subset = df[df["category"] == category_name]
    if subset.empty:
        raise ValueError(f"No rows found for category: {category_name}")
    return subset.sample(n=min(n, len(subset)), random_state=42)


def main() -> None:
    input_path = DATA_DIR / "amazon_reviews_cleaned.csv"
    output_path = DATA_DIR / "amazon_reviews_ready.csv"

    print("Loading full dataset...")
    df = pd.read_csv(input_path, low_memory=False, dtype={"vote": "str"})

    print("\nRaw dataset")
    print("===============")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nCategory counts:")
    print(df["category"].value_counts())
    print("\nSentiment counts:")
    print(df["sentiment"].value_counts())
    print("\nEmpty clean_text rows:")
    print((df["clean_text"].fillna("").str.strip() == "").sum())

    df = df.dropna(subset=["overall", "sentiment", "text", "clean_text"])
    df = df[df["clean_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["category", "asin", "clean_text"])

    print("\nCleaned dataset")
    print("===============")
    print("Shape:", df.shape)
    print("\nCategory x sentiment table:")
    print(pd.crosstab(df["category"], df["sentiment"]))

    balanced_parts = [
        sample_one_category(df, "Cell Phones and Accessories", SAMPLE_SIZE_PER_CATEGORY),
        sample_one_category(df, "Sports and Outdoors", SAMPLE_SIZE_PER_CATEGORY),
        sample_one_category(df, "Industrial and Scientific", SAMPLE_SIZE_PER_CATEGORY),
    ]
    balanced_df = pd.concat(balanced_parts, ignore_index=True)[FINAL_COLUMNS].copy()

    print("\nBalanced dataset")
    print("===============")
    print("Shape:", balanced_df.shape)
    print("\nCategory counts:")
    print(balanced_df["category"].value_counts())
    print("\nCategory x sentiment table:")
    print(pd.crosstab(balanced_df["category"], balanced_df["sentiment"]))
    print("\nSentiment counts:")
    print(balanced_df["sentiment"].value_counts())

    balanced_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nSaved ready dataset to: {output_path}")


if __name__ == "__main__":
    main()
