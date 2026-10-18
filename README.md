# Fitness Prediction with Logistic Regression
## Project Overview

This project applies Logistic Regression to predict whether a person is fit (1) or not fit (0) using health and lifestyle features such as age, weight, sleep hours, smoking habits, and more.

The goal of this project is to demonstrate:

- __Data cleaning and preprocessing.__
- __Handling missing values and inconsistent categories.__
- __Exploratory data analysis (EDA).__
- __Logistic regression modeling and evaluation.__

## Dataset Source

This project uses the __Fitness CLassification Dataset (Synthetic)__ by __Muhammed Derric__ on Kaggle. The dataset contains 2000 samples with a mix of __numerical and categorical features, some missing values,__ and __intentional noise__ to stimulate real-world challenges. 

## Tech Stack 
- Phyton
- Pandas, NumPy, Matlplotlib, Seaborn
- Statsmodels & Scikit-learn

## Workflow

### 1. Preprocessing
- Handled __missing values__ in _"sleep_hours"_ feature.
- Standarized __inconsistent values__ in _"smokes"_ feature.
- Checked __data distribution__ and identified __outliers__.
- Generated a __correlation heatmap__ to explore __feature relationships__.

### 2. Processing
- Defined __independet__ and __dependent__ variables.
- Trained a __logistic regression model__.
- Implemented a __custom confusion matrix function__ for __evaluation__.

### 3. Model Evaluation
Evaluated the Model with:
- __Accuracy__
- __Precision__
- __Recall__
- __F1-score__

## Visualization
- __Correlation Heatmap__ for explored feature relationship.
- __Distribution Plots__ for check data quality and outliers.
- __Custom Confusion Matrix__ for evaluated model performance.

## Result
Logistic Regression achieved on the test set:
__Accuracy__
- Train : 0.797
- Test : 0.765

__Precision__
- Train : 0.761
- Test : 0.738

__Recall__
- Train : 0.711
- Test : 0.667

__F1-Score__
- Train : 0.735
- Test : 0.701

## Key Insight
- __Smoking__ is __negatively associated__ with fitness.
- Higher sleep hours, actively index, and nutrition quality __increases__ fitness likelihood.
- __Logistic regression__ provides a __solid baseline__ for classification.

## Conclusion
This Project demonstrates how to build a classification model from a messy dataset by:
- __Cleaning and preprocessing data.__
- __Exploring distribution and correlations.__
- __Training and evaluating a logsitic regression model.__
It highlights the importance of data preparation  and evaluation metrics in building reliable machine learning models.
