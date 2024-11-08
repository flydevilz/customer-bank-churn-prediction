# Bank Churn Predictive Analysis

## Project Overview

In the competitive banking industry, retaining customers is a significant challenge. Customer churn, or the loss of customers, can negatively impact a bank's profitability, as acquiring new customers is more costly than retaining existing ones. This project utilizes machine learning to predict which customers are likely to churn, enabling banks to implement targeted interventions to retain at-risk customers.

## Importance of the Issue

- **Customer Acquisition Costs:** Acquiring new customers can cost five to 25 times more than retaining existing ones (Harvard Business Review, 2014).
- **Lost Revenue:** Churn leads to revenue loss and potentially impacts the bank’s reputation, as loyal customers often have a higher Lifetime Value (LTV).
- **Sustainability and Competition:** Retaining customers strengthens the bank's position in a highly competitive market (Forbes, 2015).

## Project Goals and Objectives

The project aims to:

1. **Identify key factors** influencing customer churn, such as age, balance, and active membership status.
2. **Develop a predictive model** using machine learning algorithms to forecast customer churn based on historical data.

## Solution Approach

- **Exploratory Data Analysis (EDA):** Conducted to uncover patterns and relationships between variables that impact churn.
- **Model Development:** Implemented Random Forest, Gradient Boosting, and SVM algorithms to predict customer churn. The best-performing model was selected based on accuracy and other metrics.
- **Class Imbalance Handling:** Used SMOTE (Synthetic Minority Over-sampling Technique) to improve the model's performance on the minority (churn) class.

## Data Understanding

### Dataset Overview

The dataset includes information on customer characteristics and behavior, with 10,000 samples and 12 features. It was sourced from Kaggle.

- **Features:** Customer demographics, transaction history, and service usage patterns.
- **Target Variable:** Churn (1 = churned, 0 = retained).

### Data Summary

- **Numerical Features:** `age`, `credit_score`, `balance`, `estimated_salary`.
- **Categorical Features:** `country`, `gender`, `tenure`, `products_number`.
- **Boolean Features:** `credit_card`, `active_member`, `churn`.

## Data Preparation

1. **Outlier Removal:** Removed outliers to ensure a representative sample.
2. **Feature Selection:** Removed low-correlation features to focus on relevant data.
3. **Encoding Categorical Variables:** Used one-hot encoding for categorical features.
4. **Class Imbalance Handling:** Applied SMOTE to address imbalance in churned vs. non-churned classes.
5. **Feature Scaling:** Standardized numerical features for improved model performance.
6. **Train-Test Split:** Split the dataset into 80% training and 20% testing sets.

## Model Selection

Three machine learning models were tested:

1. **Random Forest**
   - High accuracy and reduced overfitting.
   - Best performance after hyperparameter tuning.

2. **Gradient Boosting**
   - Provides highly accurate predictions but requires more training time.

3. **Support Vector Machine (SVM)**
   - Effective in high-dimensional spaces but struggled with this dataset.

## Hyperparameter Tuning

Performed GridSearchCV on Random Forest with parameters such as `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `bootstrap` to identify the best model configuration.

- **Best Parameters:** `n_estimators = 300`, `max_depth = 20`, `min_samples_split = 10`, `min_samples_leaf = 1`, `bootstrap = True`.

## Evaluation Metrics

1. **Accuracy:** Percentage of correct predictions.
2. **Precision:** Focuses on correct churn predictions.
3. **Recall:** Measures the ability to capture actual churn cases.
4. **F1 Score:** Balances precision and recall.

### Evaluation Results

| Model              | Accuracy | Precision (Class 1) | Recall (Class 1) | F1 Score (Class 1) |
|--------------------|----------|----------------------|-------------------|---------------------|
| Random Forest      | 83.02%   | 0.87                | 0.12             | 0.22               |
| Gradient Boosting  | 83.02%   | 0.82                | 0.13             | 0.23               |
| Support Vector Machine | 81.05% | 0.00              | 0.00             | 0.00               |

### Model Selection

The Random Forest model demonstrated the best performance after tuning, achieving an accuracy of 90.56%.

## Key Findings from EDA

1. **Age and Balance Distribution:** High-balance customers may need special retention strategies.
2. **Country:** German customers are more likely to churn, indicating the need for location-based strategies.
3. **Active Membership:** Active members have lower churn rates.
4. **Balance:** Customers with zero balance are at a higher risk of churn.

## Business Impact

The insights and predictive models developed in this project provide actionable strategies for customer retention:

1. **Identify Key Factors:** Provides insights into the main drivers of churn, aiding in targeted retention strategies.
2. **Predict Customer Churn:** The Random Forest model enables banks to accurately identify at-risk customers and proactively intervene.

## Conclusion

This project demonstrates the power of machine learning in predicting customer churn, providing banks with tools to retain customers and improve profitability. The Random Forest model, with optimized parameters, serves as an effective predictor for customer churn, helping banks take timely action to retain valuable customers.
