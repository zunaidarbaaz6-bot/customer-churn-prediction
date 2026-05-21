Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using real world data. The goal is to identify customers who are likely to cancel their subscription so the business can take action to retain them.


Project Overview

Churn means a customer stopped using a service. In the telecom industry losing customers is costly so companies use machine learning to predict who is at risk of leaving before it happens. This project builds an end to end pipeline from raw data to a trained model and an interactive dashboard.


Dataset

Source: Telco Customer Churn dataset from Kaggle
Records: 7043 customers
Features: 20 columns including contract type, tenure, monthly charges, payment method and internet service
Target: Churn column indicating whether a customer left or stayed


Tools and Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Power BI


Project Steps

Step 1 Data Cleaning and Preparation
Loaded the dataset and fixed the TotalCharges column which had hidden blank spaces causing conversion errors. Filled missing values with the median. Dropped the customerID column as it has no predictive value. Converted the Churn column from Yes and No to 1 and 0 for the model.

Step 2 Exploratory Data Analysis
Analyzed churn patterns across contract type, tenure, monthly charges and payment method. Key findings are listed below.

Step 3 Model Building
Encoded all categorical columns using Label Encoding. Split the data 80 percent for training and 20 percent for testing using stratified sampling. Trained and compared three models: Logistic Regression, Random Forest and XGBoost. Evaluated each model using Accuracy, F1 Score and AUC-ROC.

Step 4 Dashboard
Built an interactive dashboard in Power BI using Python visuals showing churn by contract type, churn by tenure, average monthly charges by churn status and churn rate by payment method.


Key Findings

Month to month customers churn at 43 percent compared to 3 percent for two year contract customers. Most churned customers leave within the first 5 months of joining. Churned customers pay higher average monthly charges at around 74 dollars compared to 61 dollars for customers who stayed. Electronic check users have the highest churn rate among all payment methods.


Model Results

Logistic Regression: Accuracy 79.6 percent, F1 Score 0.587, AUC-ROC 0.839
Random Forest: Accuracy 78.8 percent, F1 Score 0.553, AUC-ROC 0.826
XGBoost: Accuracy 78.9 percent, F1 Score 0.575, AUC-ROC 0.820

Best Model: Logistic Regression with AUC-ROC of 0.84


How to Run

1. Clone this repository
2. Install the required libraries using pip install pandas numpy matplotlib seaborn scikit-learn xgboost
3. Download the dataset from Kaggle and place it in the project folder
4. Run eda.py first for exploratory analysis
5. Run churn_model.py for model training and evaluation


File Structure

eda.py runs the exploratory data analysis and saves all charts
churn_model.py trains and evaluates all three models and saves the confusion matrix, ROC curve and feature importance chart
dashboard images folder contains all saved plots from the analysis


Author
Zunaid Arbaaz

Master of Applied Computing, University of Windsor
