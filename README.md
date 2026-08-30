# Google Play Store Review Sentiment Model

A machine learning pipeline that predicts whether a Google Play Store review is positive (rating ≥ 4) or not, using engagement and review-content signals rather than the rating itself.

## What it predicts

The model classifies each review as **positive (1)** or **neutral/negative (0)** based on the review's star rating (`score >= 4`). It does **not** use the rating itself as an input - only signals that are available independently of the label, so the model has to learn genuine patterns rather than just reading off the answer.

## Features used

| Feature             | Description                                      |
| ------------------- | ------------------------------------------------ |
| `thumbsUpCount`     | Number of "helpful" votes the review received    |
| `review_length`     | Character length of the review text              |
| `has_reply`         | Whether a developer replied to the review        |
| `engagement_score`  | Derived: `thumbsUpCount + (has_reply * 5)`       |
| `word_count`        | Number of words in the review                    |
| `exclamation_count` | Number of `!` characters in the review           |
| `caps_ratio`        | Proportion of uppercase characters in the review |

## Approach

1. **Data cleaning** - deduplicated by review ID, handled missing values, normalized text and numeric fields (`main.py::clean_data`).
2. **Feature engineering** - derived `review_length`, `has_reply`, and a composite `engagement_score` from raw review fields, plus `word_count`, `exclamation_count`, and `caps_ratio` from the review text itself.
3. **Model training** - a Random Forest Classifier (200 estimators) trained via a Scikit-Learn pipeline with median imputation, using an 80/20 train-test split stratified by class (`train_model.py`).
4. **Evaluation** - classification report and ROC AUC computed on the held-out test set.

## Tech stack

- **Python**
- **Pandas / NumPy** - data cleaning and feature engineering
- **Scikit-Learn** - pipeline, imputation, Random Forest Classifier, evaluation metrics
- **Joblib** - model persistence

## Results

- Test-set accuracy: **70%**
- ROC AUC: **0.73**
- Precision/recall: 0.68 / 0.69 for negative reviews, 0.72 / 0.71 for positive reviews (960 held-out samples)

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Expects a `play_store_reviews.csv` file in the working directory with at minimum: `reviewId`, `content`, `score`, `thumbsUpCount`, `replyContent`, `reviewCreatedVersion`. Outputs a cleaned dataset, a trained model (`playstore_review_model.joblib`), and a predictions file.

## Possible extensions

- Add NLP-based text features (e.g., sentiment polarity, topic modeling) beyond length/engagement signals
- Compare Random Forest against other classifiers (XGBoost, logistic regression) as a baseline
- Use SHAP to explain which features drive individual predictions
