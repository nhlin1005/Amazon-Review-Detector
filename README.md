Amazon Review Sentiment and Credibility Analyzer

Project Description
===================

This project analyzes Amazon reviews with two connected tasks.

1. Sentiment classification
The sentiment model predicts whether a review is negative, neutral, or positive.

2. Credibility and suspiciousness analysis
The system gives each review a credibility score and explains why a review may
look suspicious, such as duplicate-like wording, repetitive text, very short
generic praise, or mismatch between the star rating and the review text.

Important note:
This project does not prove that a review is fake or AI-generated.
It only estimates whether a review looks low-credibility or suspicious.


Datasets Used
=============

Training source categories:
- Cell Phones and Accessories
- Industrial and Scientific
- Sports and Outdoors

These Amazon category datasets are cleaned, combined, and then split into:
- training set
- validation set
- test set

Main split files:
- dataset\amazon_reviews_train.csv
- dataset\amazon_reviews_val.csv
- dataset\amazon_reviews_test.csv

Additional dataset for new testing:
- dataset\new_amazon_test_ready.csv

This new test dataset is created from a new Amazon category file after running
the data preparation script.


How to Run the Project
======================

Step 1. Preprocess the raw Amazon review data

Command:
python textPreprocess_v2.py

Output:
- cleaned review text
- processed dataset files used for later steps


Step 2. Check that the dataset is ready

Command:
python dataset_ready_check_v2.py

Output:
- dataset check results
- confirmation that required columns are present


Step 3. Split the dataset into train, validation, and test sets

Command:
python split_dataset_v2.py

Output:
- dataset\amazon_reviews_train.csv
- dataset\amazon_reviews_val.csv
- dataset\amazon_reviews_test.csv


Step 4. Train the sentiment model

For the current neural-network version, run:

Command:
python train_neural_network_model.py

Output:
- validation results for each MLP config
- best config selection
- final test metrics
- dataset\best_single_review_model_nn.joblib


Step 5. Build the combined credibility model

Command:
python build_combined_credibility_model.py

Output:
- dataset\combined_review_credibility_model.joblib


Step 6. Score reviews for suspiciousness

Command:
python suspicious_review_flagger_v2.py

Output:
- flagged review results
- suspiciousness summary files


Step 7. Prepare a new Amazon dataset for testing

Command:
python prepare_new_amazon_test_data.py

Output:
- dataset\new_amazon_test_ready.csv


Step 8. Apply the combined model to the new dataset

Command:
python apply_combined_credibility_model.py

Output:
- dataset\new_amazon_test_with_credibility.csv
- dataset\new_amazon_test_credibility_summary.csv


Notebook Files
==============

You can also open these notebooks to view results and examples:
- 404projecct.ipynb

