import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score
)
from xgboost import XGBClassifier

plt.style.use("seaborn-v0_8-whitegrid")

# load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# this column has some blank spaces that mess things up, so we force it to be a number
# anything that cant be converted becomes NaN, then we fill it with the middle value
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# customer ID is just a unique ID, it has nothing to do with predicting churn so we remove it
# we check first if the column actually exists before trying to remove it
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# convert Yes/No to 1/0 because the model only understands numbers
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("Data loaded. Shape:", df.shape)

# save a copy of the data before encoding so Power BI shows real names not numbers
# this must be done here before we convert anything to numbers below
df_powerbi = df.copy()
df_powerbi["Contract"] = df_powerbi["Contract"].astype(str)
df_powerbi["Churn"] = df_powerbi["Churn"].astype(str)
df_powerbi["gender"] = df_powerbi["gender"].astype(str)
df_powerbi["PaymentMethod"] = df_powerbi["PaymentMethod"].astype(str)
df_powerbi["InternetService"] = df_powerbi["InternetService"].astype(str)
df_powerbi.to_csv("churn_powerbi.csv", index=False)
print("Power BI file saved as churn_powerbi.csv")

# find all the text columns because ML models cant read text, only numbers
cat_cols = df.select_dtypes(include="object").columns.tolist()
print("Encoding", len(cat_cols), "categorical columns:", cat_cols)

# go through each text column and convert it to numbers
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("Encoding done.")

# X is everything we use to predict, y is what we are trying to predict (churn)
X = df.drop(columns=["Churn"])
y = df["Churn"]

print("Features:", X.shape[1], "| Samples:", X.shape[0])
print("Churn rate:", round(y.mean() * 100, 1), "%")

# split data into training and testing
# 80% to train the model, 20% to test how well it learned
# stratify makes sure both splits have the same churn ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# scaling brings all numbers to the same range so no single column dominates
# logistic regression needs this, tree models like random forest and xgboost dont
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# we are testing 3 different models to see which one performs best
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

results = {}

print("MODEL RESULTS")

for name, model in models.items():
    # logistic regression uses scaled data, the other two work fine without scaling
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # calculate how good the model is
    acc = accuracy_score(y_test, y_pred)   # how many it got right overall
    f1 = f1_score(y_test, y_pred)          # better metric when data is imbalanced
    auc = roc_auc_score(y_test, y_proba)   # how well it separates churned vs stayed

    results[name] = {"model": model, "y_pred": y_pred,
                     "y_proba": y_proba, "acc": acc, "f1": f1, "auc": auc}

    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("F1 Score:", round(f1, 4))
    print("AUC-ROC:", round(auc, 4))
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

# pick the model with the highest AUC score
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print("Best Model:", best_name, "AUC =", round(best["auc"], 4))

# confusion matrix shows how many predictions were correct and where it went wrong
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"], ax=ax)
ax.set_title("Confusion Matrix " + best_name, fontsize=13, fontweight="bold")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("plot_07_confusion_matrix.png", dpi=150)
plt.show()

# ROC curve shows all 3 models in one chart, higher and more left = better
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#fc8d62", "#66c2a5", "#8da0cb"]

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, label=name + " AUC=" + str(round(res["auc"], 3)), color=color, lw=2)

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("ROC Curve All Models", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plot_08_roc_curve.png", dpi=150)
plt.show()

# feature importance tells us which columns had the most impact on the prediction
xgb_model = results["XGBoost"]["model"]
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top15 = importances.sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(9, 6))
sns.barplot(x=top15.values, y=top15.index, palette="coolwarm_r", ax=ax)
ax.set_title("Top 15 Feature Importances XGBoost", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig("plot_09_feature_importance.png", dpi=150)
plt.show()

# print final numbers, copy these into your resume
print("FINAL SUMMARY")
for name, res in results.items():
    print(name, "| Accuracy:", round(res["acc"] * 100, 1), "| F1:", round(res["f1"], 3), "| AUC:", round(res["auc"], 3))

print("Best Model:", best_name)
print("AUC-ROC:", round(best["auc"], 4))
print("F1 Score:", round(best["f1"], 4))
print("Accuracy:", round(best["acc"] * 100, 2), "%")
