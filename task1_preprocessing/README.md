# Task 1: Data Cleaning & Preprocessing

**AI & ML Internship — Elevate Labs**

## Objective
Learn how to clean and prepare raw data for Machine Learning.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Dataset
**Titanic Dataset**, loaded directly via `seaborn.load_dataset("titanic")` (891 rows, 15 columns originally). A raw copy is saved at `data/titanic_raw.csv`.

## Project Structure
```
task1_preprocessing/
├── data/
│   ├── titanic_raw.csv          # original, unmodified dataset
│   └── titanic_cleaned.csv      # final cleaned & preprocessed dataset
├── images/
│   ├── boxplots_before_outlier_removal.png
│   └── boxplots_after_outlier_removal.png
├── notebook/
│   └── Task1_Data_Cleaning_Preprocessing.ipynb   # executed notebook with all outputs
├── preprocessing.py              # standalone script version of the pipeline
└── README.md
```

## What I Did (Step by Step)

### 1. Imported the dataset and explored basic info
Loaded the Titanic dataset and checked `.shape`, `.dtypes`, `.describe()`, and `.isnull().sum()` to understand the structure and identify data quality issues. Key finding: `age` had 177 missing values (~20%), `deck` had 688 missing (~77%), and `embarked`/`embark_town` each had 2 missing values.

### 2. Handled missing values
- Dropped `deck` (77% missing — too sparse to impute reliably) and redundant duplicate columns (`embark_town`, `alive`, `class`, `who`, `adult_male`, `alone`) that repeat information already present in other columns.
- **`age`** → imputed with the **median** (28.0), since age is slightly right-skewed and median is more robust to outliers than the mean.
- **`embarked`** → imputed with the **mode** (`'S'`), the standard approach for categorical columns with very few missing values.

### 3. Encoded categorical features
- **`sex`** → **Label Encoding** (`female`=0, `male`=1), appropriate since it's a binary category.
- **`embarked`** → **One-Hot Encoding** (`embarked_Q`, `embarked_S`, with `C` as the dropped baseline), appropriate since it has 3 unrelated categories with no natural order.

### 4. Normalized/standardized numerical features
Applied **`StandardScaler`** (z-score standardization, mean=0, std=1) to `age`, `fare`, `sibsp`, and `parch`. Standardization was chosen over min-max normalization because these features aren't uniformly bounded and several (like `fare`) are skewed with outliers, which standardization handles more gracefully.

### 5. Visualized and removed outliers
Used boxplots to visually inspect all four numeric columns (`images/boxplots_before_outlier_removal.png`), then removed outliers using the **IQR method** (1.5×IQR rule). This was applied only to `age` and `fare`, the two genuinely continuous variables:

| Column | Outliers removed | Reason |
|---|---|---|
| `age` | 66 rows | Continuous — IQR method appropriate |
| `fare` | 107 rows | Continuous — IQR method appropriate |
| `sibsp` | *(not removed)* | Discrete count variable, mostly 0/1 — IQR would strip nearly all non-zero values |
| `parch` | *(not removed)* | Same reasoning as `sibsp` |

**Dataset size: 891 rows → 718 rows** after outlier removal. Final result saved to `data/titanic_cleaned.csv` and visualized in `images/boxplots_after_outlier_removal.png`.

## What I Learned
- How to diagnose data quality issues (nulls, dtypes, skew) before touching the data.
- Mean vs. median vs. mode imputation, and when each is appropriate.
- The difference between label encoding (ordinal/binary) and one-hot encoding (nominal, multi-category).
- Why standardization is often preferred over normalization for skewed, unbounded features.
- That the IQR method for outlier removal is not universally appropriate — it works well on continuous variables but can wrongly flag most of a discrete/count variable as "outliers."

---

## Interview Questions & Answers

**1. What are the different types of missing data?**
- **MCAR (Missing Completely At Random):** the missingness has no relationship to any observed or unobserved data — pure chance (e.g., a sensor randomly drops a reading).
- **MAR (Missing At Random):** the missingness is related to other *observed* variables, not the missing value itself (e.g., older passengers being less likely to report age, but explainable by class or ticket type in the data).
- **MNAR (Missing Not At Random):** the missingness is related to the *value itself* (e.g., people with very high incomes deliberately not disclosing them).

**2. How do you handle categorical variables?**
Convert them into numeric form so ML algorithms can use them: **label encoding** for ordinal or binary categories, **one-hot encoding** for nominal categories with no order, or more advanced methods (target/frequency encoding, embeddings) for high-cardinality categorical features.

**3. What is the difference between normalization and standardization?**
**Normalization** (min-max scaling) rescales values into a fixed range, typically [0, 1]. **Standardization** (z-score scaling) rescales values to have a mean of 0 and standard deviation of 1, without bounding them to a fixed range. Standardization is generally preferred when data has outliers or isn't uniformly distributed; normalization is useful when a bounded range is required (e.g., neural network inputs, image pixel values).

**4. How do you detect outliers?**
Common methods: **boxplots/IQR** (values beyond 1.5×IQR from Q1/Q3), **Z-score** (values beyond ~3 standard deviations from the mean), **visual inspection** (scatter plots, histograms), and **model-based methods** (Isolation Forest, DBSCAN, Local Outlier Factor) for multivariate outliers.

**5. Why is preprocessing important in ML?**
Raw data is almost always messy — missing values, inconsistent scales, categorical text, noise, and outliers. Most ML algorithms assume clean, numeric, appropriately scaled input. Poor preprocessing leads to biased models, numerical instability, and misleading performance metrics; good preprocessing directly improves model accuracy, training stability, and generalization.

**6. What is one-hot encoding vs label encoding?**
**Label encoding** assigns each category an integer (0, 1, 2, …) — simple, but implies a false ordinal relationship if the categories aren't actually ordered. **One-hot encoding** creates a separate binary (0/1) column for each category — avoids the false-ordering problem but increases dimensionality, especially with many categories.

**7. How do you handle data imbalance?**
Techniques include: **resampling** (oversampling the minority class with SMOTE, or undersampling the majority class), **class weighting** (penalizing misclassification of the minority class more heavily in the loss function), using **evaluation metrics** beyond accuracy (precision, recall, F1, ROC-AUC) that aren't misleading on imbalanced data, and **ensemble methods** designed for imbalance (e.g., balanced random forests).

**8. Can preprocessing affect model accuracy?**
Yes, significantly. Poor handling of missing values can introduce bias; unscaled features can dominate distance-based or gradient-based algorithms (KNN, SVM, neural networks, gradient descent in general); improper encoding can create false relationships; and unaddressed outliers can skew model coefficients and predictions. Careful preprocessing is often responsible for a larger accuracy gain than model/hyperparameter tuning.

---

## Submission Notes
- Time window followed: task completed within the 10:00 AM–10:00 PM window as instructed.
- Self-research was used to review imputation strategies and reconfirm the IQR outlier method.
- All errors during development were debugged independently.
- No paid tools were used — Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn are all free/open-source.
