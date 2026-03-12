import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input_path = r"dataset\amazon_reviews_ready.csv"
    train_path = r"dataset\amazon_reviews_train.csv"
    val_path = r"dataset\amazon_reviews_val.csv"
    test_path = r"dataset\amazon_reviews_test.csv"

    df = pd.read_csv(input_path, low_memory=False)

    print("ready dataset shape:", df.shape)

    train_df, temp_df = train_test_split(df, test_size=0.30,random_state=42, stratify=df["sentiment"])

    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["sentiment"])

    print("train shape:", train_df.shape)
    print("val shape:", val_df.shape)
    print("test shape:", test_df.shape)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("due")


if __name__ == "__main__":
    main()