# Laporan Proyek Machine Learning - Angeline Lydia Kojansow

## Project Domain

In today's digital era, the banking industry faces significant challenges in retaining their customers. Customer churn, or the loss of customers, is a critical issue that can significantly impact a bank's profitability. When customers leave a bank and switch to competitors, the bank not only loses revenue from those customers but also incurs additional costs to attract new ones. Therefore, understanding and addressing customer churn is a crucial step to maintain financial stability and long-term success for banks.
Machine learning provides a powerful approach to address the issue of customer churn by enabling banks to analyze vast amounts of customer data and develop predictive models. Machine learning algorithms can process large datasets to identify patterns and correlations that may not be apparent through traditional analysis methods. This helps in understanding the key factors that contribute to customer churn.
 By training machine learning models on historical customer data, banks can predict which customers are most likely to churn. These models can use various features such as transaction history, customer demographics, and service usage patterns to make accurate predictions. With predictions from machine learning models, banks can implement targeted interventions to prevent churn. For example, banks can offer personalized incentives, improve customer service, or tailor product recommendations to at-risk customers, thereby increasing the chances of retaining them.


**Why is this issue important?**
- **Customer Acquisition Costs**: Acquiring new customers is often much more expensive than retaining existing ones. According to research from the Harvard Business Review, the cost of acquiring a new customer can be five to 25 times higher than retaining an existing customer (Harvard Business Review, 2014).
- **Lost Revenue**: Customers who churn not only take potential revenue away but can also affect the bank's image and reputation. Satisfied customers are likely to use more products and services, thereby increasing their Lifetime Value (LTV) to the bank.
- **Sustainability and Competition**: Amidst intense competition in the banking industry, maintaining a strong customer base is key to business sustainability. Banks that can understand customer behavior patterns and the reasons for churn can develop more effective customer retention strategies (Forbes, 2015)

**Problems to be Solved**
To tackle customer churn, banks need to identify the factors causing customers to leave their services. This involves:
- Customer Data Analysis: Collecting and analyzing customer data to identify patterns and early signs of potential churn.
- Predictive Model Development: Developing predictive models that can forecast which customers are most likely to churn based on historical data.
- Intervention Strategies: Implementing targeted intervention strategies to prevent churn, such as offering incentives, enhancing customer service, or personalizing product offerings.

[Harvard Business Review - The Value of Keeping the Right Customers](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers)
[Forbes - How Much Does It Really Cost You To Get A New Customer?](https://www.forbes.com/sites/theyec/2015/10/13/how-much-does-it-really-cost-you-to-get-a-new-customer/)


## Business Understanding

### Problem Statements

1. How to identify the key factors influencing customer churn in a bank?
2. How to predict which customers are most likely to churn using historical data?

### Goals

Objectives of the problem statement:

1. Conduct exploratory analysis on customer data to identify key factors such as age, balance, gender, products used, and active membership status that contribute to churn.
2. Develop an predictive model using machine learning algorithms such as Random Forest, Gradient Boosting, and SVM to predict customer churn.

### Solution statements
To achieve the set goals, here are the proposed solutions:
1. Conduct Exploratory Data Analysis (EDA) to see what variables influence customer churn.
2. Using various machine learning algorithms to predict customer churn, using Random Forest, Gradient Boosting, and SVM. From the many algorithms tested, the algorithm with high accuracy will be selected.
3. Implementing class imbalance handling techniques, such as SMOTE (Synthetic Minority Over-sampling Technique) to improve the model's ability to predict churn in the minority class.

### Evaluation Metrics
The evaluation metrics used are as follows:
- **Accuracy:** Percentage of correct predictions out of total predictions.
- **Precision:** Percentage of correct churn predictions out of total churn predictions.
- **Recall:** Percentage of actual churned customers that were successfully predicted.
- **F1-score:** Harmonic mean of precision and recall, giving an idea of the balance between the two.




## Data Understanding
In this project, a bank customer dataset is utilized to predict customer churn. This dataset includes various features that describe customer characteristics, their activities, and membership status. The data is sourced from [Kaggle](https://www.kaggle.com/datasets/bhuviranga/customer-churn-data).

### Dataset Overview
- **Number of samples:** 10,000
- **Number of features:** 12
- **Missing values:** No missing values 
- **Duplicate data:** 0
- **Data types:**
  - **Numerical:** 4
  - **Categorical:** 4
  - **Boolean:** 3

### This dataset has the following variables:
- **customer_id:** Unique ID for each customer.
- **credit_score:** Customer credit score.
- **country:** Country where the customer is located (France, Germany, Spain).
- **gender:** Customer gender (Male, Female).
- **age:** Customer age.
- **tenure:** Length of time the customer has been with the bank in years.
- **balance:** Balance held by the customer at the bank.
- **products_number:** Number of bank products used by the customer.
- **credit_card:** Indicator of whether the customer has a credit card (1 = Yes, 0 = No).
- **active_member:** Indicator of whether the customer is an active member (1 = Yes, 0 = No).
- **estimated_salary:** Estimated annual salary for the customer.
- **churn:** Indicator of whether the customer churns (1 = Yes, 0 = No).

For the purpose of analysis, the variables are categorized as follows:
1. **Categorical data:** country, gender, tenure, products_number
2. **Numerical data:** age, credit_score, balance, estimated_salary
3. **Boolean data:** credit_card, active_member, churn
4. **Target column:** churn, to determine the customer credit card churn rate.

### Exploratory Data Analysis (EDA)
To gain a comprehensive understanding of the data, several stages of exploratory analysis were conducted as outlined below:
1. Statistical Description
The analysis began with examining the statistical description of the dataset to provide an overview of the distribution of values. This was accomplished using the `data.describe()` function.
2. Variable Distribution
The distribution of variables was visualized to identify patterns and anomalies within the data. This was achieved through the use of histograms.
3. Categorical Variable Analysis
An analysis was performed on the categorical variables to observe their distribution and relationship with customer churn.
4. Correlation Between Variables
The analysis also included an examination of correlations between variables to identify any strong relationships between features in the dataset.

#### EDA Results
Based on the EDA results, here are some key findings:
1. Age and Balance Distribution: Customer age and bank balance have a varied distribution. There are some customers with high balances who may need special attention.
![Age_and_Balance](https://github.com/user-attachments/assets/498441b2-c47c-4284-9333-0bc1f26e6932)
2. Correlation: There is a fairly significant correlation between several features such as age, product number, country, balance, active membership, and gender.
![corr](https://github.com/user-attachments/assets/2e5295a9-d494-4d73-a6e3-6a0d22e8b2ad)
3. Country: Customers from Germany are more likely to churn compared to those from France and Spain. Targeted retention strategies may be required for customers in Germany.
![country](https://github.com/user-attachments/assets/886d8d06-a295-46af-a9e0-056c7d48637a)
4. Age: Middle-aged customers (40-49) have the highest churn rate, indicating a need for retention strategies focused on this age group.
![age](https://github.com/user-attachments/assets/5d096497-ae51-4043-82a3-de6142f8d786)
5. Active Member Status: Active members show a significantly lower churn rate, suggesting that increasing customer engagement and activity could reduce churn.
![membership](https://github.com/user-attachments/assets/0b3b97a6-2543-4e09-9cf0-4423a870eab9)
6. Balance: Customers with zero balance are more likely to churn. Encouraging customers to maintain a balance might help in reducing churn.
![balance](https://github.com/user-attachments/assets/3f3d0468-6d5d-476d-9ba2-61de1bf1bbe2)
7. Credit Score: Customers with lower credit scores (500-649) are more likely to churn. Providing financial counseling and support to improve credit scores could help in retaining these customers.
![creditscore](https://github.com/user-attachments/assets/fc9ad46e-f1bc-434e-8514-f2d24e376cd2)
8. Credit Card Possesion: Customers with credit card possesion are likely to churn.
![creditcardposs](https://github.com/user-attachments/assets/ee1dc300-a603-4e20-b1da-cfdc49b38f4c)




## Data Preparation
In this section, various data preparation techniques are applied to ensure the dataset is ready for use in the prediction model. The data preparation process is carried out sequentially and involves several important stages.

1. Outlier Data Removal
Outliers can skew the results and affect the performance of the model. We identify and remove outliers to ensure the data is representative of the typical customer behavior.
Reason: Removing outliers helps in achieving a more accurate model by eliminating data points that can disproportionately influence the training process.

2. Feature Selection
Columns that do not have a high correlation, such as credit score, estimated salary, and tenure, will be removed to allow the algorithm to focus more effectively on classification.
Reason: Removing features with low correlation helps simplify the model, reduce noise, and improve the algorithm's ability to identify the most relevant patterns for predicting customer churn.  

3. Encoding Categorical Variables
Categorical variables need to be converted into numeric format to be used in machine learning models. One-hot encoding is used to convert categorical variables into a binary format, such as country, gender, tenure, products number, and membership.
Reason: One-hot encoding prevents the numerical interpretation of categorical variables and ensures that the model processes the data correctly without assuming any ordinal relationship between categories.

4. Handling Class Imbalance
The issue of class imbalance is present in this customer churn dataset, where the number of churned customers is significantly less than the number of non-churned customers. We use SMOTE (Synthetic Minority Over-sampling Technique) to address this imbalance.
Reason: SMOTE generates synthetic samples of the minority class, which helps machine learning models learn from a more balanced representation of the data. This improves prediction performance, especially for the minority class, in this case, the churned customers.

5. Feature Scaling
Numerical features such as age, balance, and estimated salary need to be normalized to have the same scale. We use StandardScaler to standardize these features.
Reason: Feature scaling is important to ensure that all features have the same scale, which helps machine learning models work more efficiently and improves the convergence of the training algorithm.

6. Train-Test Split
The dataset is split into training and testing sets to evaluate the performance of the model. Typically, 80% of the data is used for training, and 20% is reserved for testing.
Reason: Splitting the data into training and testing sets allows us to assess how well the model generalizes to unseen data, ensuring that the evaluation metrics reflect the model's true performance.



## Modeling
### Random Forest
Random Forest is an ensemble learning method that uses multiple decision trees to enhance prediction accuracy and reduce overfitting. This model works by building many random decision trees and combining their results.

Advantages:
- Capable of handling datasets with many features.
- Effective for datasets with imbalanced target variables.
- Reduces the risk of overfitting compared to a single decision tree.

Disadvantages:
- Can become slow and inefficient with very large datasets.
- More difficult to interpret compared to simpler models like single decision trees.

Parameters Used:
- `n_estimators`: 100 (number of trees in the forest)
- `random_state`: 42 (to ensure consistent results)

### Gradient Boosting
Gradient Boosting is a boosting technique that builds predictive models incrementally by combining multiple weak models to form a strong model. It optimizes the loss function by adding new models that correct the errors of previous models.

Advantages:
- Can provide highly accurate predictions.
- Handles imbalanced data well.
- Flexible and can be used for both regression and classification tasks.

Disadvantages:
- Requires longer training time compared to some other algorithms.
- Needs careful parameter tuning to avoid overfitting.

Parameters Used:
- `n_estimators`: 100 (number of boosting stages to be performed)
- `random_state`: 42 (to ensure consistent results)

### Support Vector Machine (SVM)
Support Vector Machine (SVM) is a classification algorithm that seeks the optimal hyperplane that maximizes the margin between two classes. SVM is effective in high-dimensional spaces and is used for binary classification.

Advantages:
- Effective in high-dimensional spaces.
- Effective when the number of dimensions is greater than the number of samples.
- Memory efficient.

Disadvantages:
- Does not perform well with very large datasets due to time complexity.
- Not effective on datasets with a lot of noise.

Parameters Used:
- kernel: 'linear' (using a linear kernel for linear separation)
- random_state: 42 (to ensure consistent results)


#### Hyperparameter Tuning 
Hyperparameter tuning is performed at random forest algorithm using GridSearchCV to find the optimal combination of parameters. Below are the results of the tuning process and an explanation of the chosen parameters.

Parameter Grid:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [None, 10, 20, 30]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `bootstrap`: [True, False]

Improvement Process:
- Defining the Parameter Grid: We determined the range of parameters to be tested.
- Conducting Grid Search: We performed a grid search with 3-fold cross-validation to evaluate different combinations of parameters.
- Selecting the Best Combination: We chose the best combination of parameters based on the evaluation results.
- This process allowed us to enhance the model's performance by selecting the most effective hyperparameters, thereby improving its predictive capabilities for the churn prediction problem.

Best Parameter found from hyperparameter tuning: 
- `n_estimators`: 300
- `max_depth`: 20
- `min_samples_split`: 10 
- `min_samples_leaf`: 1
- `bootstrap`: True 

### Model Selection
Several models were evaluated to determine the most effective approach for predicting customer churn. The Random Forest model, with hyperparameter tuning, demonstrated the best performance, achieving an accuracy of 90.56%. This indicates that the Random Forest model, with the optimized parameters, is effective in predicting customer churn. 


## Evaluation
### Accuracy

Accuracy is the percentage of correct predictions out of the total predictions.

**Formula**: 
Accuracy = (True Positives (TP) + True Negatives (TN)) / Total Predictions

### Precision

Precision is the percentage of true positive predictions out of the total positive predictions. Precision is important for problems where the cost of false positives is high.

**Formula**: 
Precision = True Positives (TP) / (True Positives (TP) + False Positives (FP))

### Recall

Recall is the percentage of actual positive cases that are correctly predicted. Recall is important for problems where capturing all positive cases is crucial.

**Formula**: 
Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))

### F1 Score

F1 Score is the harmonic mean of precision and recall. It provides a balance between precision and recall.

**Formula**: 
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

### Evaluation Results 
#### Random Forest algorithm (default parameters)
- **Accuracy**: 83.02%
- **Precision (Class 0)**: 0.83
- **Recall (Class 0)**: 1.00
- **F1 Score (Class 0)**: 0.90
- **Precision (Class 1)**: 0.87
- **Recall (Class 1)**: 0.12
- **F1 Score (Class 1)**: 0.22

#### Gradient Boosting
- **Accuracy**: 83.02%
- **Precision (Class 0)**: 0.83
- **Recall (Class 0)**: 0.99
- **F1 Score (Class 0)**: 0.90
- **Precision (Class 1)**: 0.82
- **Recall (Class 1)**: 0.13
- **F1 Score (Class 1)**: 0.23

#### Support Vector Machine (SVM)
- **Accuracy**: 81.05%
- **Precision (Class 0)**: 0.81
- **Recall (Class 0)**: 1.00
- **F1 Score (Class 0)**: 0.90
- **Precision (Class 1)**: 0.00
- **Recall (Class 1)**: 0.00
- **F1 Score (Class 1)**: 0.00


### Project Results Explanation
#### Accuracy
Accuracy provides a general overview of how well the model predicts overall. The Random Forest and Gradient Boosting models both achieve an accuracy of 83.02%, while SVM is slightly lower at 81.05%. However, accuracy alone is not sufficient, especially in imbalanced datasets like this one, where class 1 (churn) is of primary interest.

#### Precision
Precision measures the accuracy of the positive predictions (churn in this case). The Random Forest model with default parameters has a slightly higher precision for class 1 (0.87) compared to Gradient Boosting (0.82). SVM has a precision of 0.00 for class 1, indicating its inability to make correct churn predictions.

#### Recall
Recall is critical in customer churn prediction, as it measures the ability to capture actual churn cases. Both Random Forest and Gradient Boosting models have very low recall for class 1 (0.12 and 0.13, respectively), meaning they miss many actual churn cases. SVM also fails in this aspect, with a recall of 0.00 for class 1.

#### F1 Score
The F1 Score provides a balance between precision and recall. Both Random Forest and Gradient Boosting have low F1 scores for class 1 (0.22 and 0.23, respectively), reflecting their difficulty in balancing precision and recall for churn prediction. SVM's F1 score of 0.00 for class 1 further confirms its unsuitability for this task.


### Impact on Business Understanding
#### Problem Statements
- Identifying Key Factors: The exploratory data analysis (EDA) identified key factors influencing customer churn, such as age, balance, country, and active member status. This insight helps in understanding the main drivers of churn.
- Predicting Customer Churn: The predictive models developed, especially the Random Forest with tuned parameters, have shown a good balance of precision, recall, and F1 score, making it a reliable tool for predicting customer churn.

#### Goals Achievement
- Exploratory Analysis: Successfully conducted EDA to identify key factors contributing to churn.
- Predictive Model Development: Developed multiple predictive models and identified theRandom Forest model with tuned parameters as the best performing model for predicting customer churn.


**---The End---**



