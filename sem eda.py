
# ============================================================
# PROJECT: EXPLORATORY DATA ANALYSIS OF STUDENT SEMESTER RESULTS
# Description: Analysing student performance using Python
# ============================================================

# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ------------------------------------------------------------
# 1. SETTING UP THE PROJECT
# ------------------------------------------------------------

# Reading the CSV file from the same folder as this Python file

BASE_DIR = Path(__file__).resolve().parent

FILE_PATH = BASE_DIR / "semester_results_clean.csv"

# Creating a folder to save the output graphs

OUTPUT_DIR = BASE_DIR / "EDA_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Setting the graph style

sns.set_theme(style="whitegrid", context="notebook")

print("=" * 60)
print(" EXPLORATORY DATA ANALYSIS OF STUDENT SEMESTER RESULTS")
print("=" * 60)


# ------------------------------------------------------------
# 2. LOADING THE DATASET
# ------------------------------------------------------------

if not FILE_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Please keep semester_results_clean.csv "
        "in the same folder as this Python file."
    )

df = pd.read_csv(FILE_PATH)

print("\n1. DATASET INFORMATION")

print("\nDataset Shape:", df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())


# ------------------------------------------------------------
# 3. CHECKING REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "student_id",
    "batch",
    "semester_no",
    "result",
    "percentage",
    "failed_subjects"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"The following required columns are missing: {missing_columns}"
    )


# ------------------------------------------------------------
# 4. BASIC DATA CLEANING
# ------------------------------------------------------------

# Converting numerical columns into numeric format

numeric_columns = [
    "batch",
    "semester_no",
    "percentage",
    "failed_subjects"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Removing records without essential information

df = df.dropna(
    subset=[
        "student_id",
        "batch",
        "semester_no",
        "result"
    ]
).copy()

# Converting batch and semester into integer format

df["batch"] = df["batch"].astype(int)
df["semester_no"] = df["semester_no"].astype(int)

# Standardising result labels

df["result"] = (
    df["result"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\nDataset cleaning completed.")

print("Updated Dataset Shape:", df.shape)


# ------------------------------------------------------------
# 5. STUDENTS PER BATCH AND SEMESTER
# ------------------------------------------------------------

print("\n2. STUDENTS PER BATCH AND SEMESTER")

batch_semester = pd.crosstab(
    df["batch"],
    df["semester_no"]
)

print(batch_semester)

plt.figure(figsize=(10, 6))

sns.heatmap(
    batch_semester,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Number of Student Records per Batch and Semester")
plt.xlabel("Semester Number")
plt.ylabel("Batch")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_batch_semester.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 6. RESULT DISTRIBUTION
# ------------------------------------------------------------

print("\n3. RESULT DISTRIBUTION")

result_counts = df["result"].value_counts()

print(result_counts)

print("\nResult Percentage:")

print(
    (result_counts / len(df) * 100).round(2)
)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="result",
    order=["FCD", "FC", "SC", "FAIL"],
    hue="result",
    palette={
        "FCD": "#2a9d8f",
        "FC": "#e9c46a",
        "SC": "#f4a261",
        "FAIL": "#e76f51"
    },
    legend=False
)

plt.title("Distribution of Student Results")
plt.xlabel("Result Category")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_result_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 7. PASS AND FAIL ANALYSIS
# ------------------------------------------------------------

print("\n4. PASS AND FAIL ANALYSIS")

# Calculating pass/fail directly from the result column

df["pass_status"] = np.where(
    df["result"] == "FAIL",
    "Fail",
    "Pass"
)

pass_fail = df["pass_status"].value_counts()

print("\nPass and Fail Counts:")
print(pass_fail)

print("\nPass and Fail Percentage:")

print(
    (pass_fail / len(df) * 100).round(2)
)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="pass_status",
    hue="pass_status",
    palette={
        "Pass": "#2a9d8f",
        "Fail": "#e76f51"
    },
    legend=False
)

plt.title("Student Pass and Fail Distribution")
plt.xlabel("Student Status")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_pass_fail.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 8. PERCENTAGE DISTRIBUTION
# ------------------------------------------------------------

print("\n5. PERCENTAGE DISTRIBUTION")

print(
    df["percentage"].describe().round(2)
)

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="percentage",
    bins=30,
    kde=True,
    color="#457b9d"
)

plt.title("Distribution of Student Semester Percentage")
plt.xlabel("Percentage")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_percentage_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 9. FAILED SUBJECTS ANALYSIS
# ------------------------------------------------------------

print("\n6. FAILED SUBJECTS ANALYSIS")

failed_counts = (
    df["failed_subjects"]
    .value_counts()
    .sort_index()
)

print(failed_counts)

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df.dropna(subset=["failed_subjects"]),
    x="failed_subjects",
    color="#e76f51"
)

plt.title("Number of Failed Subjects per Semester Record")
plt.xlabel("Number of Failed Subjects")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_failed_subjects.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 10. BACKLOGS VS STUDENT PERFORMANCE
# ------------------------------------------------------------

print("\n7. BACKLOGS VS STUDENT PERFORMANCE")

backlog_performance = (
    df.groupby("failed_subjects")["percentage"]
    .mean()
    .round(2)
)

print("\nAverage Percentage Based on Failed Subjects:")

print(backlog_performance)

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df.dropna(
        subset=["failed_subjects", "percentage"]
    ),
    x="failed_subjects",
    y="percentage",
    color="#a8dadc"
)

plt.title("Student Percentage Based on Failed Subjects")
plt.xlabel("Number of Failed Subjects")
plt.ylabel("Semester Percentage")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_backlogs_vs_performance.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 11. SEMESTER-WISE PERFORMANCE ANALYSIS
# ------------------------------------------------------------

print("\n8. SEMESTER-WISE AVERAGE PERFORMANCE")

semester_performance = (
    df.groupby("semester_no")["percentage"]
    .mean()
    .round(2)
)

print(semester_performance)

trend = (
    df.groupby("semester_no")["percentage"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

sns.lineplot(
    data=trend,
    x="semester_no",
    y="percentage",
    marker="o",
    linewidth=2.5,
    color="#2a9d8f"
)

plt.title("Average Student Percentage Across Semesters")
plt.xlabel("Semester Number")
plt.ylabel("Average Percentage")

plt.xticks(sorted(df["semester_no"].unique()))

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_semester_performance.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 12. BATCH-WISE PERFORMANCE
# ------------------------------------------------------------

print("\n9. BATCH-WISE AVERAGE PERFORMANCE")

batch_performance = (
    df.groupby("batch")["percentage"]
    .mean()
    .round(2)
)

print(batch_performance)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="batch",
    y="percentage",
    hue="batch",
    estimator="mean",
    errorbar=None,
    palette="viridis",
    legend=False
)

plt.title("Average Academic Performance by Batch")
plt.xlabel("Batch")
plt.ylabel("Average Percentage")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_batch_performance.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 13. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n10. CORRELATION ANALYSIS")

correlation_columns = [
    "semester_no",
    "batch",
    "percentage",
    "failed_subjects"
]

correlation = df[
    correlation_columns
].corr(method="spearman")

print(correlation.round(2))

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title("Correlation Between Academic Performance Variables")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_correlation_heatmap.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 14. SAVING ANALYSIS RESULTS
# ------------------------------------------------------------

# Saving the cleaned dataset

df.to_csv(
    OUTPUT_DIR / "cleaned_semester_results.csv",
    index=False
)

# Saving semester performance summary

semester_performance.to_csv(
    OUTPUT_DIR / "semester_performance.csv"
)

# Saving batch performance summary

batch_performance.to_csv(
    OUTPUT_DIR / "batch_performance.csv"
)


# ------------------------------------------------------------
# 15. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 60)

print("EXPLORATORY DATA ANALYSIS COMPLETED SUCCESSFULLY")

print("=" * 60)

print("\nTotal Students:", df["student_id"].nunique())

print("Total Records:", len(df))

print("Total Batches:", df["batch"].nunique())

print("Total Semesters:", df["semester_no"].nunique())

print("\nAll graphs and analysis files are saved in:")

print(OUTPUT_DIR.resolve())

print("\nThank you! EDA process completed.")
