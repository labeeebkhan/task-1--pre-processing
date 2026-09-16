"""
Task 1: Data Cleaning & Preprocessing
AI & ML Internship - Elevate Labs

Dataset : Titanic Dataset
Steps followed (per the task mini-guide):
  1. Import the dataset and explore basic info (nulls, data types)
  2. Handle missing values using mean/median/imputation
  3. Convert categorical features into numerical using encoding
  4. Normalize/standardize the numerical features
  5. Visualize outliers using boxplots and remove them
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set_style("whitegrid")

# ---------------------------------------------------------------------
# STEP 1: Import the dataset and explore basic info (nulls, data types)
# ---------------------------------------------------------------------
print("=" * 70)
print("STEP 1: IMPORT & EXPLORE THE DATASET")
print("=" * 70)

df = sns.load_dataset("titanic")
df.to_csv("data/titanic_raw.csv", index=False)

print(f"\nShape of dataset: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistical summary:")
print(df.describe(include="all").T)

print("\nMissing values per column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
print(missing_summary[missing_summary["missing_count"] > 0])

# ---------------------------------------------------------------------
# STEP 2: Handle missing values using mean/median/imputation
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: HANDLE MISSING VALUES")
print("=" * 70)

# Drop columns that are almost entirely empty or redundant/duplicate info
# 'deck' is ~77% missing -> drop
# 'embark_town' duplicates 'embarked'; 'alive' duplicates 'survived';
# 'class' duplicates 'pclass'; 'who'/'adult_male' duplicate age/sex info.
# We drop the redundant duplicates but keep the core numeric/categorical
# features so encoding & scaling has meaningful columns to work on.
cols_to_drop = ["deck", "embark_town", "alive", "class", "who", "adult_male", "alone"]
df_clean = df.drop(columns=cols_to_drop)
print(f"\nDropped high-missing / redundant columns: {cols_to_drop}")

# Age (numeric, some skew) -> impute with MEDIAN
median_age = df_clean["age"].median()
df_clean["age"] = df_clean["age"].fillna(median_age)
print(f"Filled 'age' missing values with median = {median_age}")

# Embarked (categorical, only 2 missing) -> impute with MODE
mode_embarked = df_clean["embarked"].mode()[0]
df_clean["embarked"] = df_clean["embarked"].fillna(mode_embarked)
print(f"Filled 'embarked' missing values with mode = '{mode_embarked}'")

print("\nMissing values after imputation:")
print(df_clean.isnull().sum())

# ---------------------------------------------------------------------
# STEP 3: Convert categorical features into numerical using encoding
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: ENCODE CATEGORICAL FEATURES")
print("=" * 70)

# 'sex' -> binary -> Label Encoding (male/female -> 0/1)
le = LabelEncoder()
df_clean["sex"] = le.fit_transform(df_clean["sex"])
print(f"\nLabel encoded 'sex': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 'embarked' -> more than 2 categories, no ordinal relationship -> One-Hot Encoding
df_clean = pd.get_dummies(df_clean, columns=["embarked"], prefix="embarked", drop_first=True)
new_ohe_cols = [c for c in df_clean.columns if c.startswith("embarked_")]
df_clean[new_ohe_cols] = df_clean[new_ohe_cols].astype(int)
print(f"One-hot encoded 'embarked' into: {new_ohe_cols}")

print("\nColumns after encoding:")
print(df_clean.dtypes)

# ---------------------------------------------------------------------
# STEP 4: Normalize / standardize the numerical features
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: FEATURE SCALING (STANDARDIZATION)")
print("=" * 70)

numeric_cols = ["age", "fare", "sibsp", "parch"]
scaler = StandardScaler()
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
print(f"\nStandardized columns (mean=0, std=1): {numeric_cols}")
print(df_clean[numeric_cols].describe().T[["mean", "std"]])

# ---------------------------------------------------------------------
# STEP 5: Visualize outliers using boxplots and remove them
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: OUTLIER DETECTION & REMOVAL")
print("=" * 70)

# We inspect all four numeric columns visually, but only apply IQR-based
# removal to 'age' and 'fare'. 'sibsp' and 'parch' are discrete COUNT
# variables (mostly 0/1) rather than continuous measurements, so the IQR
# method flags almost every non-zero value as an "outlier" and would strip
# away legitimate data. IQR outlier removal is best suited to continuous
# features, so age & fare are the right targets here.
visualize_cols = ["age", "fare", "sibsp", "parch"]
outlier_removal_cols = ["age", "fare"]

# --- BEFORE removal: boxplots ---
fig, axes = plt.subplots(1, len(visualize_cols), figsize=(16, 4))
for ax, col in zip(axes, visualize_cols):
    sns.boxplot(y=df_clean[col], ax=ax, color="skyblue")
    ax.set_title(f"{col} (before)")
plt.tight_layout()
plt.savefig("images/boxplots_before_outlier_removal.png", dpi=150)
plt.close()
print("\nSaved: images/boxplots_before_outlier_removal.png")

# IQR method to detect & remove outliers
def remove_outliers_iqr(data, columns):
    data = data.copy()
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(data)
        data = data[(data[col] >= lower) & (data[col] <= upper)]
        removed = before - len(data)
        print(f"  '{col}': removed {removed} outlier rows (bounds: [{lower:.2f}, {upper:.2f}])")
    return data

print("\nRemoving outliers using IQR method (age & fare only):")
df_final = remove_outliers_iqr(df_clean, outlier_removal_cols)
print(f"\nRows before outlier removal: {len(df_clean)}")
print(f"Rows after outlier removal:  {len(df_final)}")

# --- AFTER removal: boxplots (all 4 columns, to show the effect) ---
fig, axes = plt.subplots(1, len(visualize_cols), figsize=(16, 4))
for ax, col in zip(axes, visualize_cols):
    sns.boxplot(y=df_final[col], ax=ax, color="lightgreen")
    ax.set_title(f"{col} (after)")
plt.tight_layout()
plt.savefig("images/boxplots_after_outlier_removal.png", dpi=150)
plt.close()
print("Saved: images/boxplots_after_outlier_removal.png")

# ---------------------------------------------------------------------
# Save final cleaned dataset
# ---------------------------------------------------------------------
df_final.to_csv("data/titanic_cleaned.csv", index=False)
print("\n" + "=" * 70)
print(f"Final cleaned dataset saved to data/titanic_cleaned.csv  (shape: {df_final.shape})")
print("=" * 70)
