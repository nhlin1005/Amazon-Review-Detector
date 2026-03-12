import pandas as pd


def main():
    input_path = r"dataset\amazon_reviews_cleaned.csv"
    output_path = r"dataset\amazon_reviews_ready.csv"

    print("Loading full dataset...")
    df = pd.read_csv(
        input_path,
        low_memory=False,
        dtype={"vote": "str"}
    )

    print("\nraw dataset")
    print("\n===============")
    print("shape:", df.shape)

    print("\ncolumns:")
    print(df.columns.tolist())

    print("\nmissing values:")
    print(df.isnull().sum())

    print("\ncategory counts:")
    print(df["category"].value_counts())

    print("\nsentiment counts:")
    print(df["sentiment"].value_counts())

    print("\nfind the empty rows in clean_text:")
    print((df["clean_text"].fillna("").str.strip() == "").sum())

    print("\nclean the dataset")
    df = df.dropna(subset=["overall", "sentiment", "text", "clean_text"])
    df = df[df["clean_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["category", "asin", "clean_text"])

    print("\ncleaned dataset")
    print("\n===============")
    print("shape:", df.shape)

    print("\ncolumns:")
    print(df.columns.tolist())

    print("\nmissing values:")
    print(df.isnull().sum())

    print("\ncategory counts:")
    print(df["category"].value_counts())

    print("\ncategory x sentiment table:")
    print(pd.crosstab(df["category"], df["sentiment"]))

    print("sampling the dataset")
    sample_size = 40000

    cell_df = df[df["category"] == "Cell Phones and Accessories"].sample(n=min(sample_size, len(df[df["category"] == "Cell Phones and Accessories"])), random_state=42)

    sports_df = df[df["category"] == "Sports and Outdoors"].sample( n=min(sample_size, len(df[df["category"] == "Sports and Outdoors"])), random_state=42)

    industrial_df = df[df["category"] == "Industrial and Scientific"].sample(n=min(sample_size, len(df[df["category"] == "Industrial and Scientific"])), random_state=42)

    b_df = pd.concat([cell_df, sports_df, industrial_df], ignore_index=True)

    print("\nbalanced dataset")
    print("\n===============")
    print("shape:", df.shape)

    print("\ncolumns:")
    print(df.columns.tolist())

    print("\nmissing values:")
    print(df.isnull().sum())

    print("\ncategory counts:")
    print(df["category"].value_counts())

    print("\ncategory x sentiment table:")
    print(pd.crosstab(df["category"], df["sentiment"]))

    final_columns = ["category", "asin", "overall", "sentiment", "verified", "vote", "unixReviewTime", "text", "clean_text"]

    balanced_df = b_df[final_columns].copy()

    balanced_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nsaved ready dataset to:", output_path)
    print("\nfinal shape:", balanced_df.shape)

    print("\nhead:")
    print(balanced_df.head())


if __name__ == "__main__":
    main()