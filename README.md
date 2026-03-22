# Amazon Review Detector Using Machine Learning

This project builds an Amazon review analysis system that combines sentiment classification with low-credibility review detection.

The system is designed to help answer two practical questions when reading Amazon reviews:

1. What sentiment does the review express?
2. Does the review look potentially low-credibility or suspicious?

The project does not claim to prove that a review is fake. Instead, it estimates the possibility that a review may be low-credibility based on review text, metadata, sentiment behavior, and interpretable suspiciousness signals.

================================================================================
1. PROJECT OVERVIEW
================================================================================

The project has two connected parts.

Part 1: Sentiment Classification
The sentiment model predicts whether a review is:
- negative
- neutral
- positive

This part helps summarize what the review is saying.

Part 2: Low-Credibility / Suspicious Review Detection
The credibility module checks whether a review shows warning signs such as:
- repetitive wording
- duplicate-like text
- very short generic praise
- rating-text mismatch
- weak metadata patterns such as not verified purchase

This part helps estimate whether a review may be low-credibility.

By combining both parts, the final system can describe:
- the sentiment of the review
- the suspiciousness / credibility level of the review
- the reasons why the review was flagged

================================================================================
2. FINAL OUTPUT OF THE SYSTEM
================================================================================

For each review, the final combined system can output:

- predicted_sentiment
- low_credibility_score
- low_credibility_label
- suspicious_label
- suspicious_reasons

This means the project can take a processed review dataset as input and return a credibility estimate for each review.

================================================================================
3. FINAL RESULT SUMMARY
================================================================================

Best final sentiment model:
- Test Accuracy: 0.9176
- Test Macro-F1: 0.7352

The combined credibility model can process a new dataset and assign:
- a low-credibility score
- a risk label
- a suspicious label
- explanations for the flag

Example low-credibility labels:
- low
- medium
- high

Example suspicious labels:
- suspicious
- not_suspicious

================================================================================
4. REPOSITORY LAYOUT
================================================================================

Typical layout:

project/
├── 404projecct.ipynb                     # Main notebook / final presentation notebook
├── textPreprocess_v2.py                 # Raw Amazon review preprocessing
├── dataset_ready_check_v2.py            # Dataset sanity checks
├── split_dataset_v2.py                  # Train / val / test split
├── train_logistic_regression_v3.py      # Logistic Regression baseline
├── train_linear_svm_v2.py               # Linear SVM baseline
├── train_best_single_model_v3_fixed.py  # Final sentiment model training
├── utils_eval_v2.py                     # Shared evaluation helpers
├── credibility_model.py                 # Combined sentiment + credibility logic
├── build_combined_credibility_model.py  # Build final combined model bundle
├── apply_combined_credibility_model.py  # Apply combined model to a new processed dataset
├── suspicious_review_flagger_v2.py      # Run combined scorer on the main dataset
├── prepare_new_amazon_test_data.py      # Convert a new Amazon JSON file into a test-ready CSV
├── README.md                            # Project description
├── .gitignore                           # Ignore large and temporary files
└── dataset/
    ├── combined_review_credibility_model.joblib
    ├── best_single_review_model_v3.joblib
    ├── amazon_reviews_flagged_v2.csv
    ├── suspicious_review_summary_v2.csv
    ├── new_amazon_test_ready.csv
    ├── new_amazon_test_with_credibility.csv
    ├── new_amazon_test_credibility_summary.csv
    └── credibility_examples/
        ├── suspicious_examples.csv
        └── not_suspicious_examples.csv

Notes:
- The dataset folder may also contain local large CSV / JSON files that should NOT be uploaded to GitHub.
- The most important final model files are:
  - dataset\best_single_review_model_v3.joblib
  - dataset\combined_review_credibility_model.joblib

================================================================================
5. FILE-BY-FILE EXPLANATION
================================================================================

404projecct.ipynb
This is the main notebook for the project.
Use it to present:
- final sentiment metrics
- credibility scoring results
- graphs
- suspicious examples
- non-suspicious examples

textPreprocess_v2.py
This script preprocesses the original Amazon review data.
Main jobs:
- read raw review files
- combine useful text fields
- clean review text
- generate sentiment labels from star ratings
- build the project-ready dataset

dataset_ready_check_v2.py
This script checks whether the processed dataset is ready.
Main jobs:
- inspect data shape
- check columns
- check label distribution
- verify that the data looks correct before training

split_dataset_v2.py
This script splits the processed dataset into:
- training set
- validation set
- test set

train_logistic_regression_v3.py
This script trains the Logistic Regression baseline model for sentiment classification.

train_linear_svm_v2.py
This script trains the Linear SVM baseline model for sentiment classification.

train_best_single_model_v3_fixed.py
This script trains the strongest final sentiment model in the project.

utils_eval_v2.py
This file contains shared evaluation helper functions used by the training scripts.

credibility_model.py
This file contains the core logic for the combined credibility system.
Main jobs:
- load the trained sentiment model
- score suspicious patterns in the text
- combine sentiment and suspiciousness signals
- generate:
  - predicted sentiment
  - low_credibility_score
  - low_credibility_label
  - suspicious_label
  - suspicious_reasons

build_combined_credibility_model.py
This script builds and saves the final combined credibility model.
Typical output:
- dataset\combined_review_credibility_model.joblib

apply_combined_credibility_model.py
This script applies the final combined credibility model to a new processed dataset.
Main jobs:
- load a processed CSV dataset
- score each review for low credibility
- save the scored output
- save summary tables
- save suspicious and non-suspicious example files

suspicious_review_flagger_v2.py
This script runs the combined credibility scoring pipeline on the main project dataset.

prepare_new_amazon_test_data.py
This script prepares a new Amazon category dataset for testing.
Main jobs:
- load a new raw Amazon JSON review file
- clean the text
- generate the required columns
- optionally sample down to a smaller number of rows
- save a processed CSV ready for model testing

================================================================================
6. MAIN WORKFLOW
================================================================================

This section shows the normal end-to-end workflow.

Step A. Prepare the main project dataset
Use this only when you want to rebuild the original main dataset.

Run:
python textPreprocess_v2.py
python dataset_ready_check_v2.py
python split_dataset_v2.py

What this step does:
- preprocesses the raw Amazon review files
- cleans the text
- generates sentiment labels
- creates the train / validation / test CSV files

Typical outputs:
- dataset\amazon_reviews_ready.csv
- dataset\amazon_reviews_train.csv
- dataset\amazon_reviews_val.csv
- dataset\amazon_reviews_test.csv

--------------------------------------------------------------------------------
Step B. Train the sentiment models
--------------------------------------------------------------------------------

Run:
python train_logistic_regression_v3.py
python train_linear_svm_v2.py
python train_best_single_model_v3_fixed.py

What this step does:
- trains two baseline sentiment models
- trains the final stronger sentiment model
- evaluates the models
- saves the final best sentiment model bundle

Most important output:
- dataset\best_single_review_model_v3.joblib

Other typical outputs:
- result CSVs
- classification report CSVs
- confusion matrix CSVs
- search / tuning logs

--------------------------------------------------------------------------------
Step C. Build the combined credibility model
--------------------------------------------------------------------------------

Run:
python build_combined_credibility_model.py

What this step does:
- loads the trained final sentiment model
- builds the combined credibility scorer
- saves one final combined model object

Main output:
- dataset\combined_review_credibility_model.joblib

This is the packaged model used later to score datasets.

--------------------------------------------------------------------------------
Step D. Run low-credibility detection on the main dataset
--------------------------------------------------------------------------------

Run:
python suspicious_review_flagger_v2.py

What this step does:
- loads the main processed dataset
- applies the combined credibility model
- scores each review
- saves the flagged dataset and summary

Typical outputs:
- dataset\amazon_reviews_flagged_v2.csv
- dataset\suspicious_review_summary_v2.csv

Typical information produced:
- suspicious rate
- low / medium / high credibility distribution
- suspicious reasons

--------------------------------------------------------------------------------
Step E. Present the final results
--------------------------------------------------------------------------------

Open:
404projecct.ipynb

What this notebook should show:
- final sentiment accuracy and macro-F1
- confusion matrix / report summary
- credibility scoring summary
- graphs
- suspicious examples
- non-suspicious examples

This is the main notebook for presentation or final demonstration.

================================================================================
7. WORKFLOW FOR TESTING A NEW AMAZON DATASET
================================================================================

This section is for applying the model to a new Amazon category, such as:
- Arts_Crafts_and_Sewing_5.json
- Electronics_5.json
- other Amazon category review files

--------------------------------------------------------------------------------
Step 1. Prepare the new raw JSON dataset
--------------------------------------------------------------------------------

Script:
prepare_new_amazon_test_data.py

General form:
python prepare_new_amazon_test_data.py

Before running, update the settings at the top of the script, especially:
- RAW_JSON_PATH
- OUTPUT_CSV_PATH
- DATASET_DISPLAY_NAME
- MAX_ROWS (optional sample limit)

Example settings:
RAW_JSON_PATH = r"dataset\Arts_Crafts_and_Sewing_5.json\Arts_Crafts_and_Sewing_5.json"
OUTPUT_CSV_PATH = r"dataset\new_amazon_test_ready.csv"
DATASET_DISPLAY_NAME = None
MAX_ROWS = 50000

What this step does:
- loads the raw Amazon JSON file
- combines text fields
- cleans the text
- generates the required columns
- optionally samples down to a smaller size
- saves a processed CSV

Typical output:
- dataset\new_amazon_test_ready.csv

The processed CSV should contain columns such as:
- category
- asin
- overall
- sentiment
- verified
- vote
- unixReviewTime
- text
- clean_text

--------------------------------------------------------------------------------
Step 2. Apply the combined credibility model to the new dataset
--------------------------------------------------------------------------------

Script:
apply_combined_credibility_model.py

General form:
python apply_combined_credibility_model.py

This script loads:
- dataset\combined_review_credibility_model.joblib
- dataset\new_amazon_test_ready.csv

What this step does:
- scores each review in the new dataset
- predicts sentiment
- computes low_credibility_score
- assigns low_credibility_label
- assigns suspicious_label
- saves examples

Typical outputs:
- dataset\new_amazon_test_with_credibility.csv
- dataset\new_amazon_test_credibility_summary.csv
- dataset\credibility_examples\suspicious_examples.csv
- dataset\credibility_examples\not_suspicious_examples.csv

Typical summary fields:
- suspicious_rate
- high_risk_rate
- avg_score

--------------------------------------------------------------------------------
Step 3. Inspect the examples
--------------------------------------------------------------------------------

Look at:
- suspicious_examples.csv
- not_suspicious_examples.csv

These files are useful for checking whether the model output looks reasonable.

Typical suspicious examples are very short, repetitive, or generic comments such as:
- "good. good"
- "ok. ok"

================================================================================
8. ONE-SHOT QUICKSTART
================================================================================

If the main dataset and sentiment model are already prepared, the shortest workflow is:

1. Build the combined model
python build_combined_credibility_model.py

2. Run the combined model on the main dataset
python suspicious_review_flagger_v2.py

3. Prepare a new Amazon dataset
python prepare_new_amazon_test_data.py

4. Apply the combined model to the new processed dataset
python apply_combined_credibility_model.py

5. Open the main notebook
Open 404projecct.ipynb


================================================================================
9. IMPORTANT NOTE ABOUT CLAIMS
================================================================================

This project should be described as:
- an Amazon review analysis system
- a sentiment classification system
- a low-credibility / suspicious review detection system

It should NOT be described as:
- a system that proves a review is fake
- a system that proves whether a review is written by a human or not

The credibility module gives an interpretable risk estimate, not absolute proof.

================================================================================
10. AUTHOR
================================================================================

Nuohan Lin
