# Course Lead Scoring Prediction

## Overview
This repository contains a machine learning workflow for predicting course lead conversions. The project uses Logistic Regression to calculate the probability of a lead converting based on various user attributes.

## Dataset
The model is trained and evaluated on the `course_lead_scoring.csv` dataset.
*   Categorical features include `lead_source`, `industry`, `employment_status`, and `location`.
*   Numerical features include `number_of_courses_viewed`, `annual_income`, `interaction_count`, and `lead_score`.
*   The target variable to predict is `converted`.

## Data Preparation
*   Missing values in categorical columns are replaced with the string `'NA'`.
*   Missing values in numerical columns are filled with `0.0`.
*   The dataset is partitioned into training (60%), validation (20%), and test (20%) sets using a random state of 1.
*   Categorical variables are transformed via one-hot encoding using Scikit-Learn's `DictVectorizer`.

## Methodology & Evaluation
*   **Feature Importance:** Numerical features are evaluated individually using the ROC AUC score. Variables with an AUC below 0.5 are inverted to maintain a positive correlation. The `number_of_courses_viewed` feature yielded the highest AUC (0.763) among numerical features.
*   **Model Training:** A `LogisticRegression` classifier is trained on the vectorized dataset. The initial base model achieved an AUC score of roughly 0.92 on the validation set.
*   **Threshold Analysis:** Precision and recall metrics are computed across prediction thresholds ranging from 0.0 to 1.0 in 0.01 increments. The intersection of precision and recall curves occurs at a threshold of 0.55.
*   **F1 Score Calculation:** The optimal F1 score (0.881) is achieved at a decision threshold of 0.56.
*   **Cross-Validation:** The model is rigorously evaluated using 5-Fold Cross-Validation (`KFold` with `shuffle=True` and `random_state=1`). This approach resulted in a mean AUC of approximately 0.8208 with a standard deviation of 0.0288.
*   **Hyperparameter Tuning:** Tested various regularization parameter `C` values (`0.01`, `0.1`, `0.5`, `10`) using 5-Fold Cross-Validation. A `C` value of `0.01` produced the highest mean AUC score (0.827) and a standard deviation of 0.028.
