
# Customer Churn Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── 0. SETUP ────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
FIGSIZE = (10, 5)

# ── 1. LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── 2. BASIC INFO ─────────────────────────────────────────────
print("\n--- Data Types & Nulls ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fix: TotalCharges has hidden spaces → convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert target to binary
df["Churn_Binary"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\n--- Class Distribution ---")
print(df["Churn"].value_counts())
print(f"Churn Rate: {df['Churn_Binary'].mean()*100:.1f}%")

# ── 3. CHURN RATE (PIE CHART) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
df["Churn"].value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=["#66c2a5", "#fc8d62"],
    startangle=90,
    ax=ax
)
ax.set_title("Overall Churn Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("plot_01_churn_distribution.png", dpi=150)
plt.show()

# ── 4. CHURN BY CONTRACT TYPE ─────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
contract_churn = df.groupby("Contract")["Churn_Binary"].mean().reset_index()
sns.barplot(data=contract_churn, x="Contract", y="Churn_Binary", ax=ax)
ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Churn Rate")
ax.set_xlabel("Contract Type")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.tight_layout()
plt.savefig("plot_02_churn_by_contract.png", dpi=150)
plt.show()

# ── 5. CHURN BY TENURE (HISTOGRAM) ───────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
df[df["Churn"] == "Yes"]["tenure"].plot.hist(
    bins=30, alpha=0.7, label="Churned", color="#fc8d62", ax=ax
)
df[df["Churn"] == "No"]["tenure"].plot.hist(
    bins=30, alpha=0.7, label="Stayed", color="#66c2a5", ax=ax
)
ax.set_title("Tenure Distribution: Churned vs Stayed", fontsize=14, fontweight="bold")
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("plot_03_tenure_distribution.png", dpi=150)
plt.show()

# ── 6. CHURN BY PAYMENT METHOD ────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
payment_churn = df.groupby("PaymentMethod")["Churn_Binary"].mean().sort_values()
sns.barplot(x=payment_churn.values, y=payment_churn.index, ax=ax, orient="h")
ax.set_title("Churn Rate by Payment Method", fontsize=14, fontweight="bold")
ax.set_xlabel("Churn Rate")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
plt.tight_layout()
plt.savefig("plot_04_churn_by_payment.png", dpi=150)
plt.show()

# ── 7. MONTHLY CHARGES vs CHURN (BOXPLOT) ────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax,
            palette={"No": "#66c2a5", "Yes": "#fc8d62"})
ax.set_title("Monthly Charges: Churned vs Stayed", fontsize=14, fontweight="bold")
ax.set_xlabel("Churn")
ax.set_ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.savefig("plot_05_monthly_charges_boxplot.png", dpi=150)
plt.show()

# ── 8. CORRELATION HEATMAP ────────────────────────────────────
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Binary"]
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[numeric_cols].corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax
)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_06_correlation_heatmap.png", dpi=150)
plt.show()

# ── 9. KEY INSIGHTS SUMMARY ───────────────────────────────────
print("\n" + "="*50)
print("KEY EDA INSIGHTS")
print("="*50)

churn_rate = df["Churn_Binary"].mean() * 100
month_to_month_churn = df[df["Contract"] == "Month-to-month"]["Churn_Binary"].mean() * 100
avg_tenure_churned = df[df["Churn"] == "Yes"]["tenure"].mean()
avg_tenure_stayed = df[df["Churn"] == "No"]["tenure"].mean()
avg_monthly_churned = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
avg_monthly_stayed = df[df["Churn"] == "No"]["MonthlyCharges"].mean()

print(f"Overall churn rate         : {churn_rate:.1f}%")
print(f"Month-to-month churn rate  : {month_to_month_churn:.1f}%")
print(f"Avg tenure (churned)       : {avg_tenure_churned:.1f} months")
print(f"Avg tenure (stayed)        : {avg_tenure_stayed:.1f} months")
print(f"Avg monthly charge (churned): ${avg_monthly_churned:.2f}")
print(f"Avg monthly charge (stayed) : ${avg_monthly_stayed:.2f}")

