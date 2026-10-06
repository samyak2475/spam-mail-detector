# Spam Mail Detector
Internship Task 2 - Machine Learning

A text classification project that identifies whether SMS messages are spam or ham (not spam).
The project trains and evaluates Multinomial Naive Bayes and Logistic Regression models on extracted text features.

## Dataset
- Source: SMS Spam Collection (UCI Machine Learning Repository)
- Initial messages: 5,574 (4,827 ham, 747 spam, 13.40% spam)
- Duplicate messages removed: 403
- Final messages: 5,171 (4,518 ham, 653 spam, 12.63% spam)

## Approach
- Clean raw text by converting to lowercase, removing non-letter characters, and stripping extra whitespace.
- Encode labels as binary values (ham = 0, spam = 1).
- Split the dataset into train and test sets (80/20 stratified split) before feature extraction to prevent data leakage.
- Compare TF-IDF and Bag of Words representations with Multinomial Naive Bayes and Logistic Regression.

## Results
Test set metrics (1,035 messages: 904 ham, 131 spam) across all four model combinations:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Naive Bayes (TF-IDF) | 0.9623 | 0.9894 | 0.7099 | 0.8267 |
| Logistic Regression (TF-IDF) | 0.9478 | 0.9873 | 0.5954 | 0.7429 |
| Naive Bayes (Bag of Words) | 0.9787 | 0.9431 | 0.8855 | 0.9134 |
| Logistic Regression (Bag of Words) | 0.9739 | 0.9906 | 0.8015 | 0.8861 |

Multinomial Naive Bayes with Bag of Words achieved the best overall performance with a test F1 score of 0.9134.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the notebook:
   jupyter notebook spam_detector.ipynb
