# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Project Overview

This project is the first stage of an end-to-end AI/ML lifecycle. The objective of Part 1 is to acquire raw data, perform data cleaning, analyze data quality issues, explore relationships between variables, and prepare a clean dataset suitable for machine learning modeling.

The dataset selected for this project is the **House Prices - Advanced Regression Techniques** dataset from Kaggle.
The dataset contains information about residential houses, including:

* Property characteristics
* Location information
* Construction details
* Quality measurements
* Garage, basement, and outdoor features
* Sale price information

The dataset contains:

* **1460 rows**
* **81 columns**
* Numeric and categorical features
* A continuous target variable: `SalePrice`

The dataset is suitable for this capstone because it satisfies the requirement of having more than 500 records, multiple feature types, categorical variables, and a numeric prediction target.

---

# Dataset Loading and Initial Inspection

The dataset was loaded using pandas:

```python
pd.read_csv()
```

Initial inspection included:

* Displaying the first five rows
* Checking column data types
* Checking dataset dimensions
* Reviewing dataset information

The initial dataset shape was:

```
(1460, 81)
```

---

# Missing Value Analysis and Cleaning

Missing values were analyzed using:

```python
df.isnull().sum()

(df.isnull().sum()/df.shape[0])*100
```

Columns with more than 20% missing values included:

* Alley
* FireplaceQu
* PoolQC
* Fence
* MiscFeature
* Several basement and garage-related features

For this dataset, many missing values represent the absence of a feature rather than incomplete data.

Examples:

* Missing `PoolQC` means the house does not have a pool.
* Missing `GarageType` means the house does not have a garage.
* Missing `Fence` means the property has no fence.

Therefore, these categorical missing values were replaced with:


"NoFeature"


This preserves the original meaning of the data.

For numeric columns with less than 20% missing values, median imputation was applied.

The median was selected instead of the mean because:

* House price data contains extreme values and outliers.
* The mean is sensitive to extreme observations.
* The median provides a more robust estimate of central tendency.

After cleaning:

```
Total Remaining Missing Values: 0
```

---

# Duplicate Detection and Removal

Duplicate rows were checked using:

```python
df.duplicated().sum()
```

Duplicate rows were removed using:

```python
df.drop_duplicates()
```

The dataset did not contain problematic duplicate records after cleaning.

---

# Data Type Correction and Memory Optimization

Data types were reviewed using:

```python
df.dtypes
```

The column `MSSubClass` was converted from numeric representation to string because it represents a housing classification category rather than a continuous measurement.

The column `MSZoning` was converted to categorical datatype.

Memory usage comparison:

Before conversion:

```
3516.45 KB
```

After conversion:

```
3507.03 KB
```

Memory reduction:

```
9.42 KB
```

---

# Descriptive Statistics and Skewness Analysis

Descriptive statistics were generated using:

```python
df.describe()
```

Skewness was calculated for all numeric columns.

The most skewed feature identified was:

```
Column: MiscVal

Skewness:
24.4768
```

`MiscVal` has a strong positive skew.

This means:

* Most houses have very small or zero miscellaneous value.
* A small number of houses have extremely large miscellaneous values.
* The distribution contains a long right tail.

For positively skewed distributions, the mean is pulled upward by extreme values. Therefore, the median is usually a better representation of the typical value.

---

# Mean vs Median Comparison

The two most skewed numeric variables were compared using:

* Mean
* Median

Because the variables showed strong skewness, median-based imputation was selected where required.

The median provides a more representative central value when extreme observations exist.

---

# Outlier Detection Using IQR Method

Outliers were identified using:

* Q1 = 25th percentile
* Q3 = 75th percentile
* IQR = Q3 - Q1
* Lower bound = Q1 - 1.5 × IQR
* Upper bound = Q3 + 1.5 × IQR

Two variables were analyzed.

## SalePrice

Number of outliers:

```
61
```

SalePrice contains high-value properties that naturally exist in the housing market. These observations were retained because they may represent genuine expensive houses rather than data errors.

## GrLivArea

Number of outliers:

```
31
```

Large living areas may represent luxury properties. These values were retained for further analysis and will be considered during model development in Part 2.

---

# Data Visualization Analysis

## 1. Line Plot

A line plot was created using SalePrice across dataset row index.

Purpose:

* Observe price variation across records.

---

## 2. Bar Chart

Average SalePrice was compared across Neighborhood categories.

Observation:

* Different neighborhoods show significant differences in average house prices.

This indicates location is an important factor for prediction.

---

## 3. Histogram

Histogram of the most skewed feature:

```
MiscVal
```

Observation:

The distribution is highly right-skewed because most values are zero while a few properties contain large miscellaneous values.

---

## 4. Scatter Plot

Scatter plot:

```
GrLivArea vs SalePrice
```

Observation:

A positive relationship exists between above-ground living area and sale price.

Generally:

* Larger living areas correspond to higher prices.
* The relationship is not perfectly linear because other factors influence price, such as quality and location.

---

## 5. Box Plot

Box plot:

```
OverallQual vs SalePrice
```

Observation:

Higher quality ratings generally correspond to higher sale prices.

The median price increases as overall quality increases, showing that house quality is an important predictive feature.

---

# Pearson Correlation Analysis

A Pearson correlation matrix was calculated using:

```python
df.corr()
```

##In the House Prices dataset because it has many numeric columns (around 38 numeric features). When we display all correlations with annot=True, every cell shows a value, so the labels overlap and become unreadable. so make it readable by selecting the most important correlated features. Instead of: 38 x 38 = 1444 cells it shows: 15 x 15 = 225 cells. only for readability. for analysis used all parameters.

The strongest correlation was:

```
GarageCars and GarageArea

Correlation:
0.8825
```

This indicates a strong positive linear relationship.

However, correlation does not prove causation.

A possible explanation is that both variables are influenced by a third factor:

```
Overall house size
```

Larger houses often have both:

* Larger garage areas
* More garage capacity

---

# Spearman Correlation Analysis

Spearman rank correlation was calculated using:

```python
df.corr(method='spearman')
```

The largest differences between Pearson and Spearman correlations were:

| Variable Pair          | Difference |
| ---------------------- | ---------: |
| LotFrontage - LotArea  |   0.249559 |
| BedroomAbvGr - LotArea |   0.218098 |
| TotRmsAbvGrd - LotArea |   0.215909 |

These differences indicate that some relationships may be monotonic but not perfectly linear.

For Part 2 feature selection:

* Pearson correlation will be used for identifying linear relationships.
* Spearman correlation will provide additional insight into nonlinear monotonic relationships.

---

# Grouped Aggregation Analysis

The dataset was grouped by:

```
Neighborhood
```

and analyzed using:

```python
mean
std
count
```

Results:

Highest average SalePrice group:

```
NoRidge
```

Highest standard deviation group:

```
NoRidge
```

Mean price ratio:

```
3.40
```

This indicates neighborhood contains meaningful predictive information.

However, the high within-group standard deviation shows that neighborhood alone cannot fully predict house price because properties within the same area can vary significantly.

---

# Output Dataset

The cleaned dataset was saved as:

```
cleaned_data.csv
```

using:

```python
df.to_csv('cleaned_data.csv', index=False)
```

Final dataset:

```
Shape:
(1460, 81)

Missing Values:
0
```

This cleaned dataset will be used in Part 2 for supervised machine learning.

---

# Project Structure

```
part1/
│
├── data/
│   └── train.csv
│
├── plots/
│   ├── line_plot.png
│   ├── bar_chart.png
│   ├── histogram.png
│   ├── scatter_plot.png
│   ├── box_plot.png
│   └── correlation_heatmap.png
│
├── part1_eda.ipynb
│
├── cleaned_data.csv
│
└── README.md
```

---

# How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

Run the notebook:

```bash
jupyter notebook part1_eda.ipynb
```

The notebook executes all steps from data loading to final cleaned dataset generation.

---

# Conclusion

Part 1 successfully transformed raw housing data into a clean dataset through:

* Missing value treatment
* Duplicate removal
* Data type optimization
* Statistical analysis
* Outlier investigation
* Visualization
* Correlation analysis

The resulting dataset is ready for machine learning modeling in Part 2.
