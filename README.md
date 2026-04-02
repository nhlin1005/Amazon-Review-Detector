# Amazon Review Sentiment and Credibility Analyzer

## Project Description

This project analyzes Amazon reviews with two connected tasks.

**1. Sentiment Classification**
The sentiment model predicts whether a review is negative, neutral, or positive.
The final model uses a fine-tuned RoBERTa transformer (`cardiffnlp/twitter-roberta-base-sentiment`),
chosen for its strong and balanced performance across all three sentiment classes.

**2. Credibility and Suspiciousness Analysis**
The system gives each review a credibility score and explains why a review may look
suspicious, such as duplicate-like wording, repetitive text, very short generic praise,
or mismatch between the star rating and the review text.

> **Important note:**
> This project does not prove that a review is fake or AI-generated.
> It only estimates whether a review looks low-credibility or suspicious.

---

## Datasets Used

**Training source categories:**
- Cell Phones and Accessories
- Industrial and Scientific
- Sports and Outdoors

These Amazon category datasets are cleaned, combined, and then split into:
- Training set
- Validation set
- Test set

**Main split files:**
- `dataset/amazon_reviews_train.csv`
- `dataset/amazon_reviews_val.csv`
- `dataset/amazon_reviews_test.csv`

**Additional dataset for new testing:**
- `dataset/new_amazon_test_ready.csv`

This new test dataset is created from the Arts, Crafts and Sewing Amazon category
after running the data preparation script (50,000 reviews).

**Dataset split sizes:**

| Split | Rows |
|-------|------|
| Train | 96,000 |
| Validation | 18,000 |
| Test | 18,000 |
| New test (Arts, Crafts and Sewing) | 50,000 |

---

## Model Performance

### RoBERTa Sentiment Model (Final Model)

The RoBERTa model was fine-tuned for 4 epochs on a balanced subset of the training data.
Epoch 3 produced the best validation macro F1 and was selected as the final checkpoint.

**Training log:**

| Epoch | Train Loss | Val Accuracy | Val Macro F1 | Val Loss |
|-------|-----------|-------------|-------------|----------|
| 1 | 0.7113 | 0.7807 | 0.7853 | 0.5070 |
| 2 | 0.4354 | 0.8027 | 0.8015 | 0.4758 |
| **3** | **0.3501** | **0.8120** | **0.8114** | **0.5227** |
| 4 | 0.2829 | 0.8067 | 0.8078 | 0.5382 |

**Test set results (balanced: 500 per class):**

| Metric | Value |
|--------|-------|
| Accuracy | 0.8147 |
| Macro Precision | 0.8124 |
| Macro Recall | 0.8147 |
| Macro F1 | **0.8127** |
| Weighted F1 | 0.8127 |

**Per-class breakdown (test set):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.8203 | 0.8400 | 0.8300 | 500 |
| Neutral | 0.7670 | 0.6980 | 0.7309 | 500 |
| Positive | 0.8499 | 0.9060 | 0.8771 | 500 |
| **Macro avg** | **0.8124** | **0.8147** | **0.8127** | 1500 |

**Confusion matrix (test set):**

| | Pred: Negative | Pred: Neutral | Pred: Positive |
|---|---|---|---|
| **True: Negative** | 420 | 68 | 12 |
| **True: Neutral** | 83 | 349 | 68 |
| **True: Positive** | 9 | 38 | 453 |

The RoBERTa model achieves substantially more balanced performance across all three classes
compared to traditional models, which tend to underperform heavily on the neutral class.

---

### Credibility Scoring Results

**Credibility signal weights:**

| Signal | Weight |
|--------|--------|
| Duplicate / near-duplicate wording | 0.30 |
| Generic short praise | 0.22 |
| Rating–text sentiment mismatch | 0.22 |
| Very short text | 0.12 |
| High token repetition | 0.10 |
| Unverified purchase | 0.08 |

**Thresholds:**

| Label | Score Range |
|-------|-------------|
| Low credibility | < 0.30 |
| Medium credibility | 0.30 – 0.60 |
| High credibility (most suspicious) | ≥ 0.60 |
| Flagged as suspicious | score ≥ 0.45 |

**Main dataset (120,000 reviews — Cell Phones, Sports, Industrial):**

| Label | Rate |
|-------|------|
| Not suspicious | 97.92% |
| Suspicious | 2.08% |

**New test dataset (50,000 reviews — Arts, Crafts and Sewing):**
Applied using the combined credibility model (`combined_review_credibility_model.joblib`).
Credibility scores and suspicious flags are saved per-review with explanation of reasons
(e.g., `very_short_generic_text`, `high_token_repetition`, `not_verified_purchase`, `rating_text_mismatch`).

---

## Pre-trained Results

All datasets, trained models, and result files are available for download here:

**Google Drive:** https://drive.google.com/file/d/1R9PqoDBKUFDyu5t5BukqZBB5muCLtbDi/view

The archive includes:
- `dataset/amazon_reviews_train.csv`, `val.csv`, `test.csv`
- `dataset/best_single_review_model_roberta_fast.joblib` — trained RoBERTa model bundle
- `dataset/combined_review_credibility_model.joblib` — combined credibility model
- `dataset/amazon_reviews_flagged_v2.csv` — main dataset with credibility scores
- `dataset/new_amazon_test_with_credibility.csv` — new test dataset scored results
- All metric CSVs and classification reports

---

## How to Run the Project

### Step 1. Preprocess the raw Amazon review data

```bash
python textPreprocess_v2.py
```

Output: cleaned review text and processed dataset files.

### Step 2. Check that the dataset is ready

```bash
python dataset_ready_check_v2.py
```

Output: dataset check results and confirmation that required columns are present.

### Step 3. Split the dataset into train, validation, and test sets

```bash
python split_dataset_v2.py
```

Output:
- `dataset/amazon_reviews_train.csv`
- `dataset/amazon_reviews_val.csv`
- `dataset/amazon_reviews_test.csv`

### Step 4. Train the RoBERTa sentiment model

```bash
python train_roberta_model.py
```

Output:
- Validation metrics logged per epoch
- Best checkpoint selected by macro F1
- `dataset/best_single_review_model_roberta_fast.joblib`

### Step 5. Build the combined credibility model

```bash
python build_combined_credibility_model.py
```

Output: `dataset/combined_review_credibility_model.joblib`

### Step 6. Score reviews for suspiciousness

```bash
python suspicious_review_flagger_v2.py
```

Output: flagged review results and suspiciousness summary files.

### Step 7. Prepare a new Amazon dataset for testing

```bash
python prepare_new_amazon_test_data.py
```

Output: `dataset/new_amazon_test_ready.csv`

### Step 8. Apply the combined model to the new dataset

```bash
python apply_combined_credibility_model.py
```

Output:
- `dataset/new_amazon_test_with_credibility.csv`
- `dataset/new_amazon_test_credibility_summary.csv`

---

## Notebook Files

You can also open the notebooks below to view results and examples:

- `404projecct.ipynb` — main project notebook
- `project_result_checker_notebook.ipynb` — result checker: loads all datasets,
  model metrics, credibility summaries, and displays suspicious / not-suspicious examples
