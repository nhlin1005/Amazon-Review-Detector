Amazon Review Sentiment and Credibility Analyzer

Project Description
============================================================

This project analyzes Amazon reviews with two connected tasks.

First, it performs sentiment classification. The sentiment model predicts
whether a review is negative, neutral, or positive.

Second, it performs credibility and suspiciousness analysis. The system gives
each review a credibility score and explains why a review may look suspicious,
such as duplicate-like wording, repetitive text, very short generic praise,
or mismatch between the star rating and the review text.

Important note:
This project does not prove that a review is fake or AI-generated.
It only estimates whether a review looks low-credibility or suspicious.


Datasets Used for Training
============================================================

The training data comes from Amazon review datasets in these categories:

- Cell Phones and Accessories
- Industrial and Scientific
- Sports and Outdoors

These category files are processed, cleaned, combined, and then split into:

- training set
- validation set
- test set

The main split files used by the training scripts are:

- dataset\amazon_reviews_train.csv
- dataset\amazon_reviews_val.csv
- dataset\amazon_reviews_test.csv


How to Run the Project
============================================================

Step 1: Preprocess the raw Amazon review data

Command:
python textPreprocess_v2.py

This step cleans the raw review text and prepares the processed dataset.


Step 2: Check that the dataset is ready

Command:
python dataset_ready_check_v2.py

This step checks whether the dataset has the required columns and is ready
for training.


Step 3: Split the dataset into train, validation, and test sets

Command:
python split_dataset_v2.py

This step creates:

- dataset\amazon_reviews_train.csv
- dataset\amazon_reviews_val.csv
- dataset\amazon_reviews_test.csv


Step 4: Train the sentiment model

For the current neural-network version, run:

Command:
python train_neural_network_model.py

This step trains the MLP sentiment model and saves the neural-network model file.


Step 5: Build the combined credibility model

Command:
python build_combined_credibility_model.py

This step combines the sentiment model and the credibility scoring logic into
one final model file.


Step 6: Score reviews for suspiciousness

Command:
python suspicious_review_flagger_v2.py

This step applies the model to reviews and produces suspiciousness results.


Step 7: Prepare a new Amazon dataset for testing

Command:
python prepare_new_amazon_test_data.py

This step converts a new Amazon category file into the format needed by the
model.


Step 8: Apply the combined model to the new dataset

Command:
python apply_combined_credibility_model.py

This step produces the credibility and suspiciousness results for the new
dataset.


Notebook Files
============================================================

You can also open these notebooks to view results and examples:

- 404projecct.ipynb

