# ==========================================
# Stage 2: Exploratory Data Analysis (EDA)
# MovieIQ Project
# ==========================================
# ---------- Import Libraries ----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# ---------- Load Dataset ----------
df = pd.read_csv("movies.csv")

# ---------- Parse Genres ----------
def parse_genres(g):
    try:
        parsed = ast.literal_eval(g)
        return [genre["name"] for genre in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []

df["genre_list"] = df["genres"].apply(parse_genres)

df["primary_genre"] = df["genre_list"].apply(
    lambda x: x[0] if len(x) > 0 else "Unknown"
)

# ---------- Create Target Column ----------
df["success"] = (df["revenue"] > df["budget"]).astype(int)

# =====================================================
# STEP 1 : Summary Statistics
# =====================================================

print("\n========== SUMMARY STATISTICS ==========\n")
print(df.describe())

# =====================================================
# STEP 2 : Budget vs Revenue Scatter Plot
# =====================================================

print("\nCorrelation between Budget and Revenue:")
print(df["budget"].corr(df["revenue"]))

plt.figure(figsize=(8,6))
plt.scatter(df["budget"], df["revenue"], alpha=0.6)

plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.title("Budget vs Revenue")

plt.tight_layout()
plt.savefig("budget_vs_revenue.png")
#plt.show()

# =====================================================
# STEP 3 : Correlation Heatmap
# =====================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df[["budget","revenue","popularity","runtime","vote_average"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
#plt.show()

# =====================================================
# STEP 4 : Genre Count
# =====================================================

genre_counts = df["primary_genre"].value_counts()

print("\n========== PRIMARY GENRE COUNT ==========\n")
print(genre_counts)

plt.figure(figsize=(10,5))

genre_counts.plot(kind="bar")

plt.title("Movies by Primary Genre")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")

plt.tight_layout()
plt.savefig("genre_counts.png")
#plt.show()

# =====================================================
# STEP 5 : Genre Success Rate
# =====================================================

genre_success = df.groupby("primary_genre")["success"].mean() * 100

print("\n========== SUCCESS RATE BY GENRE ==========\n")
print(genre_success)

plt.figure(figsize=(10,5))

genre_success.sort_values().plot(kind="bar")

plt.title("Success Rate by Genre")
plt.xlabel("Genre")
plt.ylabel("Success Rate (%)")

plt.tight_layout()
plt.savefig("genre_success_rate.png")
#plt.show()

# =====================================================
# STEP 6 : Popularity vs Success
# =====================================================

print("\nAverage Popularity")
print(df.groupby("success")["popularity"].mean())

plt.figure(figsize=(6,5))

df.boxplot(column="popularity", by="success")

plt.title("Popularity vs Success")
plt.suptitle("")
plt.xlabel("Success")
plt.ylabel("Popularity")

plt.tight_layout()
#plt.show()

# =====================================================
# STEP 7 : Runtime vs Success
# =====================================================

print("\nAverage Runtime")
print(df.groupby("success")["runtime"].mean())

plt.figure(figsize=(6,5))

df.boxplot(column="runtime", by="success")

plt.title("Runtime vs Success")
plt.suptitle("")
plt.xlabel("Success")
plt.ylabel("Runtime")

plt.tight_layout()
#plt.show()

# =====================================================
# STEP 8 : Vote Average vs Success
# =====================================================

print("\nAverage Vote Average")
print(df.groupby("success")["vote_average"].mean())

plt.figure(figsize=(6,5))

df.boxplot(column="vote_average", by="success")

plt.title("Vote Average vs Success")
plt.suptitle("")
plt.xlabel("Success")
plt.ylabel("Vote Average")

plt.tight_layout()
#plt.show()

# =====================================================
# STAGE 2 COMPLETED
# =====================================================

# =====================================================
# STAGE 3 : Statistical Testing
# =====================================================

from scipy.stats import ttest_ind, chi2_contingency

# =====================================================
# STEP 1 : Independent T-Test
# =====================================================

successful = df[df["success"] == 1]["popularity"]
unsuccessful = df[df["success"] == 0]["popularity"]

t_stat, p_value = ttest_ind(successful, unsuccessful, equal_var=False)

print("\n========== T-TEST : Popularity vs Success ==========\n")
print(f"T-Statistic : {t_stat:.4f}")
print(f"P-Value     : {p_value:.4f}")

if p_value < 0.05:
    print("\nResult: Reject the Null Hypothesis (H₀)")
    print("Popularity differs significantly between successful and unsuccessful movies.")
else:
    print("\nResult: Fail to Reject the Null Hypothesis (H₀)")
    print("Popularity does not differ significantly between successful and unsuccessful movies.")
    # =====================================================
# STEP 2 : Chi-Square Test (Genre vs Success)
# =====================================================

# Create contingency table
contingency_table = pd.crosstab(df["primary_genre"], df["success"])

print("\n========== CONTINGENCY TABLE ==========\n")
print(contingency_table)

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\n========== CHI-SQUARE TEST ==========\n")
print(f"Chi-Square Statistic : {chi2:.4f}")
print(f"Degrees of Freedom   : {dof}")
print(f"P-Value              : {p:.4f}")

if p < 0.05:
    print("\nResult: Reject the Null Hypothesis (H₀)")
    print("Genre has a significant association with movie success.")
else:
    print("\nResult: Fail to Reject the Null Hypothesis (H₀)")
    print("Genre has no significant association with movie success.")

# ==========================================
# Stage 4: Predictive Modeling
# ==========================================

# Import additional libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# Encode primary_genre
encoder = LabelEncoder()
df["primary_genre"] = encoder.fit_transform(df["primary_genre"])
print(df["primary_genre"].head())

# Features
X = df[[
    "budget",
    "popularity",
    "runtime",
    "vote_average",
    "primary_genre"
]]

# Target
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Training samples:", X_train.shape)
print("Testing samples :", X_test.shape)

# Model Evaluation

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("========== MODEL PERFORMANCE ==========")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

# Plot
plt.figure(figsize=(8,5))
plt.bar(feature_importance["Feature"],
        feature_importance["Importance"])

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

import joblib

joblib.dump(
    {
        "model": model,
        "label_encoder": encoder,
        "feature_cols": X.columns.tolist()
    },
    "model.pkl"
)

print("Model saved successfully as model.pkl")