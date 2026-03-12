import pandas as pd
import json
import os
import re

# Step 1: Set up the file paths
cell_path = r"dataset\Cell_Phones_and_Accessories_5.json\Cell_Phones_and_Accessories_5.json"
industrial_path = r"dataset\Industrial_and_Scientific_5.json\Industrial_and_Scientific_5.json"
sports_path = r"dataset\Sports_and_Outdoors_5.json\Sports_and_Outdoors_5.json"


# Step 2: Function to load the JSON file
def load_amazon_reviews(file_path, category_name):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            review["category"] = category_name
            data.append(review)

    df = pd.DataFrame(data)
    return df

# load all categories
cell_df = load_amazon_reviews(cell_path, "Cell Phones and Accessories")
industrial_df = load_amazon_reviews(industrial_path, "Industrial and Scientific")
sports_df = load_amazon_reviews(sports_path, "Sports and Outdoors")

# check the shape
print("Cell Phones shape:", cell_df.shape)
print("Industrial shape:", industrial_df.shape)
print("Sports shape:", sports_df.shape)

# Step 3: Keep only the only needed columns
needed_cols = ["category", "asin", "reviewText", "summary", "overall", "verified", "vote", "unixReviewTime"]

def needed_columns(df):
    existing_cols = [col for col in needed_cols if col in df.columns]
    return df[existing_cols].copy()


cell_df = needed_columns(cell_df)
industrial_df = needed_columns(industrial_df)
sports_df = needed_columns(sports_df)

print("\nColumns in Cell Phones dataset:")
print(cell_df.columns.tolist())


# Step 4: Set labels based on star ratings
'''
1-2 = negative
3 = neutral
4-5 = positive
'''

def rating_to_sentiment(rating):

    if rating in [1.0, 2.0, 1, 2]:
        return "negative"
    elif rating in [3.0, 3]:
        return "neutral"
    elif rating in [4.0, 5.0, 4, 5]:
        return "positive"
    else:
        return None

for df in [cell_df, industrial_df, sports_df]:
    df["sentiment"] = df["overall"].apply(rating_to_sentiment)

print("\nSentiment distribution in Cell Phones:")
print(cell_df["sentiment"].value_counts())

# Step 5: Clean text and create one combined text column

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


for df in [cell_df, industrial_df, sports_df]:
    df["summary"] = df["summary"].fillna("")
    df["reviewText"] = df["reviewText"].fillna("")

    df["text"] = (df["summary"] + " " + df["reviewText"]).str.strip()
    df["clean_text"] = df["text"].apply(clean_text)

print("\nExample cleaned reviews:")
print(cell_df[["category", "overall", "sentiment", "text", "clean_text"]].head())

# Combine all datasets into one dataframe
all_reviews_df = pd.concat([cell_df, industrial_df, sports_df], ignore_index=True)

print("\nCombined dataset shape:", all_reviews_df.shape)
print("\nCombined category counts:")
print(all_reviews_df["category"].value_counts())

print("\nCombined sentiment counts:")
print(all_reviews_df["sentiment"].value_counts())

output_path = r"dataset\amazon_reviews_cleaned.csv"
all_reviews_df.to_csv(output_path, index=False, encoding="utf-8")
print(f"\nSaved cleaned dataset to: {output_path}")