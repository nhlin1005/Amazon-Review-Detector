# Amazon Review Detector Using Machine Learning

This project builds an Amazon review analysis system that combines sentiment classification with low-credibility review detection.

The system is designed to help users answer two practical questions when reading Amazon reviews:

1. What sentiment does the review express?
2. Does the review look potentially low-credibility or suspicious?

The project does not claim to prove that a review is fake. Instead, it estimates the possibility that a review may be low-credibility based on review text, metadata, sentiment behavior, and interpretable suspiciousness signals.

--------------------------------------------------------------------------------
PROJECT IDEA
--------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------
FINAL OUTPUT OF THE SYSTEM
--------------------------------------------------------------------------------

For each review, the final combined system can output:

- predicted_sentiment
- low_credibility_score
- low_credibility_label
- suspicious_label
- suspicious_reasons

This means the project can take a processed review dataset as input and return a credibility estimate for each review.

--------------------------------------------------------------------------------
FINAL RESULT SUMMARY
--------------------------------------------------------------------------------

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

--------------------------------------------------------------------------------
MAIN FILES IN THIS PROJECT
--------------------------------------------------------------------------------

Below is a detailed explanation of the main files.

1. 404projecct.ipynb
This is the main notebook for the project.
It can be used as the central demonstration notebook for showing:
- final sentiment results
- credibility scoring results
- graphs
- suspicious examples
- non-suspicious examples

If you want one notebook to present the final project, this is the most important one.

2. textPreprocess_v2.py
This script preprocesses the original Amazon review datasets.
Main jobs:
- read raw review files
- combine useful text fields
- clean review text
- generate sentiment labels from star ratings
- build the project-ready dataset

This script is the starting point for preparing the main data.

3. dataset_ready_check_v2.py
This script checks whether the processed dataset is ready to use.
Main jobs:
- inspect data shape
- check columns
- check label distribution
- verify that the data looks correct before training

This is mainly a validation / sanity-check script.

4. split_dataset_v2.py
This script splits the processed dataset into:
- training set
- validation set
- test set

Main jobs:
- preserve label balance
- preserve category structure
- create clean train / val / test CSV files

5. train_logistic_regression_v3.py
This script trains the Logistic Regression baseline model for sentiment classification.

Main jobs:
- build TF-IDF features
- train Logistic Regression
- evaluate on validation and test sets
- report accuracy, macro-F1, confusion matrix, and classification report

This is one of the required baseline models.

6. train_linear_svm_v2.py
This script trains the Linear SVM baseline model for sentiment classification.

Main jobs:
- build TF-IDF features
- train Linear SVM
- evaluate on validation and test sets
- save results

This is the second major baseline model.

7. train_best_single_model_v3_fixed.py
This script trains the strongest final sentiment model in the project.

Main jobs:
- run the best feature combination found during tuning
- train the final sentiment classifier
- save the final sentiment model bundle
- save results such as metrics, reports, and confusion matrix

This is the main trained sentiment model used later in the combined credibility system.

8. utils_eval_v2.py
This file contains evaluation helper functions used by the training scripts.

Main jobs:
- compute evaluation metrics
- save metrics neatly
- generate reports and confusion matrices

This file supports the training pipeline but is not usually run directly by itself.

9. credibility_model.py
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

This is the most important file for the final combined model design.

10. build_combined_credibility_model.py
This script builds and saves the final combined credibility model.

Main jobs:
- load the trained sentiment model bundle
- create the combined credibility scoring object
- save one final model file

Typical output:
- dataset\combined_review_credibility_model.joblib

This is the file that creates the final packaged credibility model.

11. apply_combined_credibility_model.py
This script applies the final combined credibility model to a new processed dataset.

Main jobs:
- load a processed CSV dataset
- score each review for low credibility
- save the scored output
- save summary tables
- save suspicious and non-suspicious example files

This is the main script to use when testing a new dataset.

12. suspicious_review_flagger_v2.py
This script runs the credibility scoring pipeline on the main project dataset.

Main jobs:
- load the main processed dataset
- apply the combined scorer
- create flagged outputs
- summarize suspicious rates and credibility levels

This is useful for generating the main suspicious-review results on the original dataset.

13. prepare_new_amazon_test_data.py
This script prepares a new Amazon dataset for testing.

Main jobs:
- load a new raw Amazon JSON review file
- clean the text
- generate the required columns
- optionally sample down to a smaller number of rows
- save a processed CSV ready for model testing

This is the script to use before applying the model to a new category.

--------------------------------------------------------------------------------
TYPICAL WORKFLOW
--------------------------------------------------------------------------------

A. Prepare the main project data
Run:
python textPreprocess_v2.py
python dataset_ready_check_v2.py
python split_dataset_v2.py

B. Train baseline and final sentiment models
Run:
python train_logistic_regression_v3.py
python train_linear_svm_v2.py
python train_best_single_model_v3_fixed.py

C. Build the final combined credibility model
Run:
python build_combined_credibility_model.py

D. Run low-credibility detection on the main dataset
Run:
python suspicious_review_flagger_v2.py

E. Test a new Amazon category
Step 1:
python prepare_new_amazon_test_data.py

Step 2:
python apply_combined_credibility_model.py

--------------------------------------------------------------------------------
INPUT FORMAT FOR A NEW PROCESSED DATASET
--------------------------------------------------------------------------------

A processed dataset should include columns such as:
- category
- asin
- overall
- sentiment
- verified
- vote
- unixReviewTime
- text
- clean_text

The most important text input for the model is:
- clean_text

--------------------------------------------------------------------------------
WHAT TO UPLOAD TO GITHUB
--------------------------------------------------------------------------------

Recommended files to upload:
- 404projecct.ipynb
- textPreprocess_v2.py
- dataset_ready_check_v2.py
- split_dataset_v2.py
- train_logistic_regression_v3.py
- train_linear_svm_v2.py
- train_best_single_model_v3_fixed.py
- utils_eval_v2.py
- credibility_model.py
- build_combined_credibility_model.py
- apply_combined_credibility_model.py
- suspicious_review_flagger_v2.py
- prepare_new_amazon_test_data.py
- README.md
- .gitignore

Recommended files NOT to upload:
- raw multi-GB datasets
- full processed CSV training data
- full flagged large CSV outputs
- __pycache__
- .ipynb_checkpoints

--------------------------------------------------------------------------------
IMPORTANT NOTE ABOUT CLAIMS
--------------------------------------------------------------------------------

This project should be described as:
- an Amazon review analysis system
- a sentiment classification system
- a low-credibility / suspicious review detection system

It should NOT be described as:
- a system that proves a review is fake
- a system that proves whether a review is written by a human or not

The credibility module gives an interpretable risk estimate, not absolute proof.

--------------------------------------------------------------------------------
AUTHOR
--------------------------------------------------------------------------------

Nuohan Lin
