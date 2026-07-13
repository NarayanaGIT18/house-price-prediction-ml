# Part 2 — Supervised Machine Learning Model

## Project Overview

This part builds supervised machine learning models using the cleaned house price dataset generated in Part 1.

Two predictive tasks were performed:

1. Regression:
   - Predict continuous house prices using SalePrice.

2. Binary Classification:
   - Classify houses into high-price and low-price categories based on the median SalePrice.

---

# Dataset Description

Dataset:
House Prices - Advanced Regression Techniques (Kaggle)

The dataset contains 1460 house records with numerical and categorical features describing house characteristics such as area, quality, rooms, garage information, and other property attributes.

The cleaned dataset from Part 1 was used as input.

---

# Target Definition

## Regression Target

The regression label was:
y_reg = SalePrice
SalePrice is a continuous numerical variable representing the final selling price of each house.

---

## Classification Target

The binary classification label was created using the median SalePrice:


y_clf = (SalePrice > SalePrice.median()).astype(int)


Definition:

- 1 → House price above median
- 0 → House price equal to or below median

Class distribution:

Training set:

- Class 1: 597
- Class 0: 571

The classes were balanced, therefore SMOTE or class weighting was not required.

---

# Data Preprocessing

## Categorical Encoding

Categorical variables were converted using one-hot encoding with:


pd.get_dummies(drop_first=True)


One-hot encoding was selected because categorical variables such as neighbourhood and material types do not have a natural numerical order.

Label encoding would incorrectly introduce an ordinal relationship between categories.

---

# Train-Test Split and Scaling

The dataset was divided using:


train_test_split(
test_size=0.2,
random_state=42
)


StandardScaler was fitted only on training data.

The scaler was not fitted on the full dataset because that would cause data leakage by allowing information from the test set to influence the training process.

---

# Regression Models

## Linear Regression

Performance:

| Model | MSE | R2 |
|---|---|---|
| Linear Regression | 6.89e9 | 0.102 |

---

## Ridge Regression

Performance:

| Model | MSE | R2 |
|---|---|---|
| Ridge Regression | 1.32e9 | 0.828 |

Ridge regression performed better because L2 regularization reduces coefficient instability caused by correlated features.

The alpha parameter controls the strength of regularization.

---

# Feature Coefficient Analysis

The three largest absolute coefficients from Linear Regression were:

| Feature | Coefficient |
|-|-|
| PoolArea | 264264.91 |
| PoolQC_NoFeature | 234612.08 |
| RoofMatl_CompShg | -120654.60 |

Since features were standardized, coefficients represent the change in predicted SalePrice for a one standard deviation increase in the feature while keeping other features constant.

Positive coefficients increase predicted price, while negative coefficients decrease predicted price.

---

# Classification Model

## Logistic Regression

Evaluation results:
Accuracy: 0.935
Precision: 0.894
Recall: 0.969
F1 Score: 0.930
AUC: 0.966


---
# Confusion Matrix


[[146 15]
[ 4 127]]


The model correctly classified most houses with only a small number of false predictions.

---

# Precision and Recall

Precision formula:
Precision = TP / (TP + FP)
Recall formula:
Recall = TP / (TP + FN)


For this problem, recall is important because missing a high-value house classification is more costly than incorrectly predicting a low-value house as high-value.

---

# ROC Curve and AUC

The ROC curve evaluates classification performance at different thresholds.

The obtained AUC:
0.9659

An AUC close to 1 indicates excellent separation between the two classes.

---

# Decision Threshold Analysis

Threshold performance:

| Threshold | Precision | Recall | F1 |
|-|-|-|-|
|0.30|0.883|0.977|0.928|
|0.40|0.889|0.977|0.931|
|0.50|0.894|0.969|0.930|
|0.60|0.906|0.954|0.929|
|0.70|0.912|0.954|0.933|

The best F1 score was obtained at:
Threshold = 0.70
F1 = 0.933

Increasing the threshold improved precision but slightly reduced recall.

---

# Regularization Experiment

Comparison:

| Model | Precision | Recall | AUC |
|-|-|-|-|
| Logistic Regression C=1.0 |0.912|0.954|0.966|
| Logistic Regression C=0.01 |0.928|0.977|0.979|

Reducing C increases regularization strength.

The C=0.01 model performed better, suggesting stronger regularization improved generalization.

---

# Bootstrap Confidence Interval

500 bootstrap samples were generated to compare:
AUC(C=1.0) - AUC(C=0.01)

Results:

Mean AUC Difference: -0.0143
95% Confidence Interval: -0.0284 to -0.0024

The confidence interval does not include zero, indicating that the improved performance of the C=0.01 model is consistent across different bootstrap samples.

---
# Files