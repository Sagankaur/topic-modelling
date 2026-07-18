"""
EDA Script for Glassdoor Review Dataset
---------------------------------------

Performs complete exploratory data analysis on:
    updated_glassdoor_data.csv

Output Includes:
    • Basic info
    • Missing values
    • Unique counts
    • Company distribution
    • Date trends
    • Rating distributions
    • Text field analysis
    • Duplicate checks
    • Outlier checks
"""

import pandas as pd
import numpy as np

# ============================================
# LOAD DATA
# ============================================
print("\n" + "="*70)
print("📂 LOADING DATA")
print("="*70)

df = pd.read_csv("updated_glassdoor_data.csv")

print("\n===== BASIC INFO =====")
print(df.info())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

# ============================================
# SHAPE
# ============================================
print("\n" + "="*70)
print("📏 DATA SHAPE")
print("="*70)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ============================================
# COLUMN OVERVIEW
# ============================================
print("\n" + "="*70)
print("📌 COLUMN NAMES")
print("="*70)

print(df.columns.tolist())

# ============================================
# MISSING VALUES
# ============================================
print("\n" + "="*70)
print("⚠️ MISSING VALUES")
print("="*70)

print("\nMissing value counts:")
print(df.isnull().sum().sort_values(ascending=False))

print("\nMissing value percentages (%):")
print((df.isnull().mean() * 100).round(2))

# ============================================
# UNIQUE COUNTS
# ============================================
print("\n" + "="*70)
print("🔎 UNIQUE VALUES")
print("="*70)

for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# ============================================
# COMPANY ANALYSIS
# ============================================
print("\n" + "="*70)
print("🏢 COMPANY ANALYSIS")
print("="*70)

print("Unique companies:", df["firm_name"].nunique())

print("\nTop 10 companies by review count:")
print(df["firm_name"].value_counts().head(10))

# ============================================
# DATE ANALYSIS
# ============================================
print("\n" + "="*70)
print("📅 DATE ANALYSIS")
print("="*70)

df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\nReviews per year:")
print(df["date"].dt.year.value_counts().sort_index())

print("\nReviews per month:")
print(df["date"].dt.to_period("M").value_counts().sort_index())

# ============================================
# RATINGS ANALYSIS
# ============================================
print("\n" + "="*70)
print("⭐ RATINGS DISTRIBUTION")
print("="*70)

rating_cols = [
    "rating",
    "Recommend",
    "CEO Approval",
    "Business Outlook",
    "Career Opportunities",
    "Compensation and Benefits",
    "Senior Management",
    "Work/Life Balance",
    "Culture & Values",
    "Diversity & Inclusion"
]

for col in rating_cols:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].describe())
        print("\nValue counts:")
        print(df[col].value_counts(dropna=False))

# ============================================
# TEXT FIELD ANALYSIS
# ============================================
print("\n" + "="*70)
print("✏️ TEXT FIELD ANALYSIS")
print("="*70)

text_cols = ["title", "pros", "cons", "job", "status"]

for col in text_cols:
    if col in df.columns:
        print(f"\n--- {col} ---")
        df[col] = df[col].astype(str)
        print("Missing:", df[col].isnull().sum())
        print("Average length:", df[col].str.len().mean())
        print("Top 10 most common values:")
        print(df[col].value_counts().head(10))

# ============================================
# FIRM LINK ANALYSIS
# ============================================
print("\n" + "="*70)
print("🔗 FIRM LINK ANALYSIS")
print("="*70)

print("Unique firm links:", df["firm_link"].nunique())

# ============================================
# DUPLICATE ANALYSIS
# ============================================
print("\n" + "="*70)
print("📎 DUPLICATE ROWS")
print("="*70)

dupes = df.duplicated().sum()
print("Duplicate rows:", dupes)

# ============================================
# OUTLIER CHECKS
# ============================================
print("\n" + "="*70)
print("🚨 OUTLIER CHECKS")
print("="*70)

print("\nRatings outside range [1–5]:")
if "rating" in df.columns:
    print(df[(df["rating"] < 1) | (df["rating"] > 5)][["rating"]])

print("\n" + "="*70)
print("🎉 EDA COMPLETE")
print("="*70)
