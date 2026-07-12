# Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

## House Price Prediction using the Ames Housing Dataset (Data from Kaggle)

## Project Overview

The objective of this project is to perform data acquisition, data cleaning, and exploratory data analysis (EDA) on the Ames Housing dataset. Before building any machine learning model, it is important to understand the dataset, identify data quality issues, and explore the relationships between variables.

The dataset was loaded into a pandas DataFrame and inspected for missing values, duplicate records, incorrect data types, skewness, and outliers. Appropriate preprocessing techniques were applied, including median imputation for selected numeric columns and data type optimization. Several visualizations were created to better understand the distribution of variables and the relationships between them. Finally, a cleaned dataset (`cleaned_data.csv`) was generated for use in Part 2 of the capstone project.
## Dataset Description

The dataset used in this project is the **Ames Housing Dataset**, which contains detailed information about residential properties in Ames, Iowa, USA. This dataset is widely used for data analysis and machine learning because it includes a combination of numerical and categorical features that influence house prices.

### Dataset Summary

* **Dataset Name:** Ames Housing Dataset
* **File Format:** CSV
* **Total Records (Rows):** 1,460
* **Total Features (Columns):** 81
* **Target Variable:** `SalePrice`
* **Numeric Features:** 38
* **Categorical Features:** 43

### Why This Dataset Was Selected

This dataset was selected because it meets all the requirements of this capstone project. It contains more than 500 records, includes both numerical and categorical variables, has a numeric target variable (`SalePrice`), and contains missing values and outliers that make it suitable for practicing data cleaning, exploratory data analysis, and machine learning techniques.
## Data Acquisition and Data Cleaning

### Data Loading

The dataset was loaded into a pandas DataFrame using the `pd.read_csv()` function. After loading, the first five rows, data types, dataset shape, and general dataset information were examined to understand the structure of the data.

### Missing Value Analysis

A missing value analysis was performed using `df.isnull().sum()` and the percentage of missing values was calculated for every column. Columns with more than 20% missing values were identified, documented and analyzed separately because they may introduce bias. These columns included:

* Alley           = 93.7% missing values
* MasVnrType      = 59.7% missing values
* FireplaceQu     = 47.2% missing values
* PoolQC          = 99.5% missing values
* Fence           = 80.7% missing values
* MiscFeature     = 96.3% missing values

For numeric columns with less than 20% missing values, the missing values were replaced using the **median** of the respective column.

**Reason for choosing the median:**
The median is less sensitive to extreme values and skewed distributions than the mean. The dataset contains skewed distributions and has outliers. Since several numerical features in this dataset are positively skewed, the median provides a more representative measure of central tendency and produces more reliable imputed values.

### Duplicate Detection

Duplicate records were checked using `df.duplicated().sum()`.

* Duplicate rows found:  **0**

Since no duplicate records were present, no rows were removed and the dataset size remained unchanged.

### Data Type Optimization

The Ames dataset may not actually contain an incorrectly inferred dtype, so here doesn't convert an incorrect dtype.

All numeric columns were correctly inferred by pandas. Therefore, no numeric dtype correction was required. The Neighborhood column was converted from object to category to improve memory efficiency.

To improve memory efficiency, repetitive text data were converted from the `object` data type to the `category` data type. The `Neighborhood` column was converted to the `category` data type because it contains a limited number of repeated values.

Memory usage was measured before and after the conversion.

* Memory before conversion: **3,600,842 bytes**
* Memory after conversion: **3,218,159 bytes**
* Memory saved: **382,683 bytes**

Converting suitable columns to the `category` data type reduced memory consumption without changing the information contained in the dataset.
## Descriptive Statistics, Skewness and Outlier Analysis

### Descriptive Statistics

Descriptive statistics were generated using the `df.describe()` function to summarize the numerical features in the dataset. The summary included the count, mean, standard deviation, minimum, maximum, and quartile values (25%, 50%, and 75%) for each numeric column. These statistics provide an overall understanding of the distribution and variability of the data.

### Skewness Analysis

The skewness of every numeric column was calculated using the `skew()` function.

The columns with the highest positive skewness were:

| Column   | Skewness |
| -------- | -------: |
| MiscVal  |    24.48 |
| PoolArea |    14.83 |
| LotArea  |    12.21 |

The **MiscVal** column had the highest absolute skewness, indicating a highly positively skewed distribution. This means that most properties have a value close to zero for miscellaneous expenses, while a small number of properties have very large values that create a long right tail.

In positively skewed data, extreme high values increase the mean more than the median. Therefore, the median is a more reliable measure of central tendency and was preferred for imputing missing values where applicable.

### Outlier Detection using the IQR Method

Outliers were identified using the Interquartile Range (IQR) method.

The following columns were analysed:

| Column    | Number of Outliers |
| --------- | -----------------: |
| SalePrice |                 61 |
| LotArea   |                 69 |

The detected outliers were **not removed** from the dataset because they may represent genuine high-value properties rather than data entry errors. Removing them could reduce valuable information that is important for predictive modelling.

Instead, the outliers were documented and will be evaluated further during Part 2. If necessary, suitable techniques such as transformation, scaling, or capping may be applied depending on the machine learning algorithm and model performance.
## Exploratory Data Analysis (Visualizations)

Several visualizations were created to understand data distributions and relationships between variables.

---
### 1. Line Plot

A line plot was created using a numeric variable to observe trends across observations.

This plot helps visualize how values change sequentially. Although the dataset is not time-series based, the line plot provides a general sense of variation across records.

---

### 2. Bar Chart

A bar chart was used to compare the **average SalePrice across different Neighborhoods**.

The results show that certain neighborhoods such as *NoRidge* and *NridgHt* have significantly higher average house prices compared to others like *MeadowV* and *IDOTRR*.

This indicates that **Neighborhood is a strong predictive feature** for house prices.

---

### 3. Histogram

A histogram was plotted for the most skewed variable (**MiscVal**).

The distribution is heavily right-skewed, meaning most values are zero while a few extreme values exist on the higher end.

This confirms the earlier skewness analysis and highlights the presence of extreme outliers in the dataset.

---

### 4. Scatter Plot

A scatter plot was created between **GrLivArea (above ground living area)** and **SalePrice**.

The plot shows a **positive relationship**, meaning that as living area increases, the house price generally increases as well.

This suggests a strong predictive relationship between these variables.

---

### 5. Box Plot

A box plot was used to visualize **SalePrice across different Neighborhoods**.

The plot shows clear differences in median house prices between neighborhoods. Some neighborhoods have higher median values and wider spread, indicating variability in property pricing.

This confirms that categorical location-based features are important for prediction.

---

### 6. Correlation Heatmap

A correlation heatmap was generated using Pearson correlation.

Strong correlations were observed between:

* GarageCars and GarageArea (0.88)
* GrLivArea and TotRmsAbvGrd (0.82)
* TotalBsmtSF and 1stFlrSF (0.82)
* OverallQual and SalePrice (0.79)

These strong correlations indicate that house size, quality, and garage capacity are important factors influencing house prices.

It was also observed that correlation does not imply causation, and some relationships may be influenced by hidden variables such as overall house size or construction quality.

For example, larger houses naturally require larger garages. Therefore, the strong correlation between GarageCars and GarageArea is likely explained by overall house size rather than one variable directly causing the other.

## Advanced Statistical Analysis

### 1. Pearson vs Spearman Correlation

Both Pearson and Spearman correlation matrices were computed to understand relationships between numerical variables.

* **Pearson correlation** measures linear relationships.
* **Spearman correlation** measures monotonic (rank-based) relationships.

It was observed that some variable pairs showed higher Spearman correlation compared to Pearson correlation, indicating **non-linear but consistent relationships**.

This suggests that some features move together in a consistent pattern, even if not in a strictly linear manner.

For feature selection in modeling, Spearman correlation will also be considered because it captures relationships that Pearson may miss.

---

### 2. Mean vs Median (Skewness Impact)

For highly skewed variables, both mean and median were compared before imputation.

It was observed that for skewed distributions such as **LotArea** and **SalePrice**, the mean is influenced by extreme values, whereas the median provides a more stable central value.

Therefore, **median was chosen for missing value imputation**, as it better represents the central tendency in skewed distributions.

---

### 3. Grouped Aggregation Analysis

A grouped analysis was performed using the **Neighborhood** feature against **SalePrice**.

The results show clear variation in house prices across neighborhoods.

* Highest average price: **NoRidge**
* Lowest average price: **MeadowV**

The ratio between highest and lowest mean prices indicates a strong predictive signal from the Neighborhood feature.

The ratio of the highest group mean to the lowest group mean is 3.40, indicating a substantial difference in average house prices across neighborhoods. This suggests that the Neighborhood feature carries useful predictive information for estimating house prices.


---

### Key Insight

Categorical features such as Neighborhood carry strong predictive power for house prices, and should be carefully encoded and used in the machine learning model in Part 2.
## Conclusion

In this project, a complete data analysis workflow was performed on the Ames Housing dataset, starting from raw data inspection to cleaned dataset creation and exploratory data analysis.

The dataset was successfully cleaned by handling missing values, optimizing data types, and removing duplicates. Exploratory analysis revealed important patterns such as the strong influence of house size, quality, and neighborhood on sale price.

Key statistical techniques such as skewness analysis, outlier detection, correlation analysis, and grouped aggregation were applied to better understand the dataset structure. Visualizations further supported these findings and provided intuitive insights into variable relationships.

The final output of this stage is a cleaned dataset (`cleaned_data.csv`) that is ready for machine learning modeling in Part 2.

---

## Key Learnings

* Data cleaning is essential before any modeling process
* Median is more reliable than mean for skewed distributions
* Neighborhood and Overall Quality are strong predictors of house price
* Correlation analysis helps identify important relationships between variables
* Visualization is critical for understanding data patterns

---

## How to Run the Project

1. Clone or download the repository.
2. Install required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Place the dataset (`train.csv`) inside the `data/` folder.
4. Open the Jupyter Notebook:

   ```text
   part1_eda.ipynb
   ```
5. Run all cells from top to bottom.
6. The cleaned dataset and plots will be generated automatically.

---

## Project Structure

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
├── cleaned_data.csv
├── part1_eda.ipynb
└── README.md
```

---

## Final Note

This completes Part 1 of the Capstone Project. The dataset has been successfully cleaned, explored, and analyzed. All key data quality issues such as missing values, duplicates, and outliers have been addressed, and important statistical patterns have been identified.

The resulting cleaned dataset (`cleaned_data.csv`) is now ready for Part 2.

