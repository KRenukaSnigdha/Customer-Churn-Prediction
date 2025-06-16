# Customer-Churn-Prediction
 Customer Churn Prediction using Supervised Learning.

This project predicts customer churn for a telecom company using a Random Forest classifier. The dataset contains customer information and whether they have churned or not.

## Dataset

The dataset is sourced from a public GitHub repository:

- [Telco Customer Churn Dataset](https://raw.githubusercontent.com/blastchar/telco-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv)

It contains customer details such as demographics, account information, and services subscribed.

## Project Overview

1. Load and preprocess the dataset:
   - Remove unnecessary columns (`customerID`).
   - Convert `TotalCharges` to numeric and handle missing values.
   - Encode categorical features into numeric labels.

2. Split the data into training and testing sets (70% train, 30% test).

3. Train a Random Forest classifier on the training data.

4. Evaluate the model on the test data using a confusion matrix and classification report.

## Dependencies

1.pandas

2.scikit-learn

## How to Run

1. Ensure you have Python installed (version 3.6+ recommended).

2. Install required libraries:
   ```bash
   pip install pandas scikit-learn

## Save your Python script as churn_prediction.py and run it.

   ```bash
   python churn_prediction.py
  ```bash

## License
This project is open-source and available under the MIT License.
