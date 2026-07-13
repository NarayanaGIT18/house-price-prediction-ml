# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## Project Overview

This part focuses on building advanced machine learning models, comparing ensemble techniques, tuning hyperparameters, creating a reproducible ML pipeline, and saving the final trained model.

The objective is to identify the most robust classification model and evaluate its ability to generalize to unseen data.

---

# 1. Decision Tree Baseline

A default Decision Tree Classifier was trained without any hyperparameter constraints (`max_depth=None`).

### Results

| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| Unrestricted Decision Tree | 1.0000 | 0.8664 |

### Overfitting Analysis

The unrestricted decision tree achieved perfect training accuracy but lower testing accuracy, indicating overfitting.

Decision trees are considered high-variance models because they split data greedily at each node and do not revisit earlier decisions. A deep tree can memorize training examples, including noise, which reduces generalization performance.

---

# 2. Controlled Decision Tree

A second decision tree was trained with:

- `max_depth=5`
- `min_samples_split=20`

### Results

| Model | Training Accuracy | Testing Accuracy |
|---|---|---|
| Controlled Decision Tree | 0.9264 | 0.8938 |

### Interpretation

The controlled tree reduced overfitting compared with the unrestricted tree.

`max_depth` limits how deep the tree can grow, reducing model complexity and variance while slightly increasing bias.

`min_samples_split` prevents the tree from creating splits using very small groups of samples, reducing splits caused by noise.

The train-test gap decreased, showing improved generalization.

---

# 3. Gini vs Entropy Comparison

Two Decision Tree models were trained using different splitting criteria.

### Results

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.8973 |
| Entropy | 0.9041 |

### Impurity Formula

### Gini Impurity
Gini = 1 - Σ(pi²)

### Entropy
Entropy = -Σ(pi log2(pi))

Gini and entropy measure how mixed the classes are inside a node.

A node with:
Gini = 0

means that the node is completely pure, and all samples belong to the same class.

Entropy works similarly, where lower entropy indicates a more pure node.

---

# 4. Random Forest Classifier

Random Forest was trained using:

- `n_estimators=100`
- `max_depth=10`
- `random_state=42`

## Performance

| Metric | Score |
|---|---|
| Training Accuracy | 0.9932 |
| Testing Accuracy | 0.9486 |
| ROC-AUC | 0.9860 |

---

# Feature Importance

The top five important features were:

| Rank | Feature | Importance |
|---|---|---|
| 1 | OverallQual | 0.063577 |
| 2 | GrLivArea | 0.062596 |
| 3 | FullBath | 0.058186 |
| 4 | YearBuilt | 0.055305 |
| 5 | GarageYrBlt | 0.044517 |

### Feature Importance Explanation

Random Forest calculates feature importance by measuring the average reduction in Gini impurity caused by each feature across all decision trees.

Feature importance differs from linear regression coefficients because coefficients represent the direction and strength of a linear relationship, while Random Forest importance measures how much a feature helps reduce classification uncertainty.

---

# Bagging Concept in Random Forest

Random Forest uses bagging (Bootstrap Aggregation) to reduce variance.

Each decision tree is trained using a bootstrap sample, which means each tree receives a random sample with replacement from the training data.

During each split, only a random subset of features is considered.

By averaging predictions from many different trees, Random Forest reduces the variance of a single deep decision tree and improves generalization.

---

# 5. Gradient Boosting Classifier

Gradient Boosting was trained with:

- `n_estimators=100`
- `learning_rate=0.1`
- `max_depth=3`
- `random_state=42`

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 0.9914 |
| Testing Accuracy | 0.9281 |
| ROC-AUC | 0.9810 |

Gradient Boosting builds trees sequentially, where each new tree attempts to correct errors made by previous trees. This allows the model to capture complex relationships.

---

# 6. Feature Ablation Study

The five least important features identified from Random Forest were:

| Feature | Importance |
|---|---|
| Heating_Grav | 0.0000 |
| Fence_MnWw | 0.0000 |
| GarageQual_Po | 0.0000 |
| MiscFeature_TenC | 0.0000 |
| SaleType_Oth | 0.0000 |

## Comparison

| Model | Test ROC-AUC |
|---|---|
| Full Random Forest | 0.9860 |
| Reduced Random Forest | 0.9856 |

### Interpretation

Removing the five least important features caused only a very small decrease in ROC-AUC.

This indicates that these features were not significantly contributing to predictions and may represent noise.

In production, reducing unnecessary features can decrease inference cost, simplify data collection, and reduce maintenance effort. However, feature removal should only be performed when performance degradation remains within an acceptable limit.

---

# 7. Cross-Validated Model Comparison

Stratified 5-fold cross-validation was performed using ROC-AUC scoring.

## Results

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.965380 | 0.011769 |
| Controlled Decision Tree | 0.905660 | 0.018189 |
| Random Forest | 0.977681 | 0.003801 |
| Gradient Boosting | 0.979529 | 0.004551 |

### Why Cross Validation?

Cross-validation provides a more reliable estimate of model performance compared with a single train-test split because the model is evaluated on multiple different subsets of data.

StratifiedKFold maintains the same class distribution in every fold, reducing evaluation bias.

---

# 8. Hyperparameter Tuning with GridSearchCV

A pipeline was created:
SimpleImputer(strategy='median')
StandardScaler()
RandomForestClassifier()

## Parameter Grid
n_estimators:
[50,100,200]

max_depth:
[5,10,None]

min_samples_leaf:
[1,5]

Total combinations:
3 × 3 × 2 = 18 models

18 × 5 folds = 90 evaluations

## Best Parameters
max_depth = 10

min_samples_leaf = 1

n_estimators = 100

## Best CV ROC-AUC
0.9779157054393458

Grid Search performs an exhaustive search over all possible combinations, providing reliable results but requiring more computation.

Randomized Search evaluates only selected combinations, making it faster for large parameter spaces but it may not always find the optimal solution.

---

# 9. Manual Learning Curve Analysis

The tuned pipeline was trained using increasing training data sizes.

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 20% | 1.000000 | 0.981258 |
| 40% | 1.000000 | 0.980531 |
| 60% | 1.000000 | 0.981751 |
| 80% | 0.999982 | 0.982079 |
| 100% | 0.999935 | 0.985269 |

### Interpretation

Training AUC remained almost perfect as the dataset size increased, showing that the model has high capacity.

Test AUC gradually improved as more training data was added, indicating that additional data may provide small performance improvements.

The model is not strongly limited by data quantity because performance has already reached a high level, but collecting additional data could provide marginal gains.

---

# 10. Model Serialization

The final tuned pipeline was saved using Joblib.

```python
import joblib

joblib.dump(best_pipeline, "best_model.pkl")
The saved model was successfully loaded and tested:
loaded_model = joblib.load("best_model.pkl")

predictions = loaded_model.predict(sample_rows)
Prediction output:
[0 0]
The saved pipeline can be reused for future predictions without retraining.
Final Model Recommendation

Based on cross-validation and test performance, Gradient Boosting and Random Forest achieved the strongest results.

Gradient Boosting achieved the highest cross-validation ROC-AUC (0.9795), while Random Forest achieved excellent test ROC-AUC (0.9860) with stable performance.

The tuned Random Forest pipeline is recommended as the final production model because it provides strong predictive performance, handles complex feature relationships, and can be reproduced using the saved pipeline.

## Final Model Comparison

| Model | CV Mean AUC | Test AUC |
|---|---|---|
| Logistic Regression | 0.965380 | - |
| Controlled Decision Tree | 0.905660 | - |
| Random Forest | 0.977681 | 0.986020 |
| Gradient Boosting | 0.979529 | 0.981047 |
| Tuned Random Forest | 0.977916 | - |

The final model provides a good balance between predictive accuracy, robustness, and deployment reliability.
