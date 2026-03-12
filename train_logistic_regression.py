import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from utils_eval import evaluate_model

def main():
    train_df = pd.read_csv(r"dataset\amazon_reviews_train.csv", low_memory=False)
    val_df = pd.read_csv(r"dataset\amazon_reviews_val.csv", low_memory=False)
    test_df = pd.read_csv(r"dataset\amazon_reviews_test.csv", low_memory=False)

    X_train = train_df["text"].fillna("")
    X_val = val_df["text"].fillna("")
    X_test = test_df["text"].fillna("")

    y_train_text = train_df["sentiment"]
    y_val_text = val_df["sentiment"]
    y_test_text = test_df["sentiment"]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    label_names = list(label_encoder.classes_)

    print("Building TF-IDF")

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        min_df=10
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    print("Train TF-IDF shape:", X_train_tfidf.shape)
    print("Val TF-IDF shape:", X_val_tfidf.shape)
    print("Test TF-IDF shape:", X_test_tfidf.shape)

    print("\nstart training Logistic Regression")

    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X_train_tfidf, y_train)

    print("\nValidation results:")
    y_val_pred = model.predict(X_val_tfidf)
    evaluate_model("Simple TF-IDF + Logistic Regression (Val)", y_val, y_val_pred, label_names)

    print("\nTest results:")
    y_test_pred = model.predict(X_test_tfidf)
    result = evaluate_model("Simple TF-IDF + Logistic Regression (Test)", y_test, y_test_pred, label_names)

    pd.DataFrame([result]).to_csv(r"dataset\simple_logistic_results.csv", index=False)
    print("\nSaved results to dataset\\simple_logistic_results.csv")


if __name__ == "__main__":
    main()