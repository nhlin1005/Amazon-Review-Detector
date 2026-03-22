from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("dataset")
RANDOM_STATE = 42


def main() -> None:
    input_path = DATA_DIR / "amazon_reviews_ready.csv"
    train_path = DATA_DIR / "amazon_reviews_train.csv"
    val_path = DATA_DIR / "amazon_reviews_val.csv"
    test_path = DATA_DIR / "amazon_reviews_test.csv"

    df = pd.read_csv(input_path, low_memory=False)
    df["stratify_key"] = df["category"].astype(str) + "__" + df["sentiment"].astype(str)

    print(f"Ready dataset shape: {df.shape}")
    print("\nCategory x sentiment before split:")
    print(pd.crosstab(df["category"], df["sentiment"]))

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["stratify_key"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["stratify_key"],
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{split_name} shape: {split_df.shape}")
        print(pd.crosstab(split_df["category"], split_df["sentiment"]))

    train_df = train_df.drop(columns=["stratify_key"])
    val_df = val_df.drop(columns=["stratify_key"])
    test_df = test_df.drop(columns=["stratify_key"])

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved train/val/test splits.")


if __name__ == "__main__":
    main()
