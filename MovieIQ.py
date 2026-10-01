"""
MovieIQ - Stage 5: Streamlit Dashboard
Run: streamlit run MovieIQ.py
"""
import ast
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="MovieIQ", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------
# DATA LOADING + PREP (cached)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    def parse_genres(g):
        try:
            parsed = ast.literal_eval(g)
            return [d["name"] for d in parsed]
        except (ValueError, SyntaxError, TypeError):
            return []

    df["genre_list"] = df["genres"].apply(parse_genres)
    df["primary_genre"] = df["genre_list"].apply(lambda x: x[0] if x else "Unknown")
    return df

@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    return bundle["model"], bundle["label_encoder"], bundle["feature_cols"]

df = load_data()
model, le, feature_cols = load_model()

# ----------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------
st.sidebar.header("Filters")
all_genres = sorted(df["primary_genre"].unique())
selected_genres = st.sidebar.multiselect("Genre", all_genres, default=all_genres)
min_vote = st.sidebar.slider("Minimum vote average", 0.0, 10.0, 0.0, 0.1)

filtered = df[df["primary_genre"].isin(selected_genres) & (df["vote_average"] >= min_vote)]

st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption("A movie is labeled **successful** when its revenue exceeds its budget.")
st.markdown(f"**{len(filtered)}** movies match the current filters "
            f"({filtered['success'].mean():.1%} successful).")

# ----------------------------------------------------------------------
# EDA CHARTS
# ----------------------------------------------------------------------
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Budget vs Revenue")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.scatterplot(data=filtered, x="budget", y="revenue", hue="success",
                     palette={0: "#d9534f", 1: "#5cb85c"}, alpha=0.6, ax=ax)
    if len(filtered):
        m = filtered["budget"].max()
        ax.plot([0, m], [0, m], "k--", lw=1, label="revenue = budget")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Success Rate by Genre")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    genre_success = filtered.groupby("primary_genre")["success"].mean().sort_values(ascending=False)
    sns.barplot(x=genre_success.values, y=genre_success.index, hue=genre_success.index,
                palette="mako", legend=False, ax=ax)
    ax.set_xlabel("Success rate")
    st.pyplot(fig)
    plt.close(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Popularity / Runtime / Vote Avg vs Success")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, colname in zip(axes, ["popularity", "runtime", "vote_average"]):
        sns.boxplot(data=filtered, x="success", y=colname, hue="success",
                    palette={0: "#d9534f", 1: "#5cb85c"}, legend=False, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col4:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    num_cols = ["budget", "revenue", "popularity", "runtime", "vote_average", "success"]
    corr = filtered[num_cols].corr() if len(filtered) > 1 else pd.DataFrame()
    if not corr.empty:
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# ----------------------------------------------------------------------
# STATISTICAL TESTS
# ----------------------------------------------------------------------
st.header("Statistical Tests")

t_col, chi_col = st.columns(2)

with t_col:
    st.subheader("T-Test: popularity vs success")
    g0 = filtered.loc[filtered["success"] == 0, "popularity"]
    g1 = filtered.loc[filtered["success"] == 1, "popularity"]
    if len(g0) > 1 and len(g1) > 1:
        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        st.write(f"t-statistic: `{t_stat:.3f}`  |  p-value: `{p_val:.4f}`")
        st.write("✅ Significant difference" if p_val < 0.05 else "❌ No significant difference")
    else:
        st.write("Not enough data in current filter to run this test.")

with chi_col:
    st.subheader("Chi-Square: genre vs success")
    if filtered["primary_genre"].nunique() > 1:
        contingency = pd.crosstab(filtered["primary_genre"], filtered["success"])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_val_chi, dof, _ = stats.chi2_contingency(contingency)
            st.write(f"chi2: `{chi2:.3f}`  |  p-value: `{p_val_chi:.4f}`")
            st.write("✅ Significant association" if p_val_chi < 0.05 else "❌ No significant association")
        else:
            st.write("Not enough variation in current filter to run this test.")
    else:
        st.write("Select at least two genres to run this test.")

# ----------------------------------------------------------------------
# LIVE PREDICTION
# ----------------------------------------------------------------------
st.header("Predict a Movie's Success")
st.write("Enter a movie's details to get a live prediction from the trained Random Forest model.")

p1, p2, p3 = st.columns(3)
with p1:
    input_budget = st.number_input("Budget ($)", min_value=1000, value=50_000_000, step=1_000_000)
    input_genre = st.selectbox("Primary Genre", sorted(le.classes_))
with p2:
    input_popularity = st.slider("Popularity", 0.0, 100.0, 50.0)
    input_runtime = st.slider("Runtime (min)", 60, 240, 120)
with p3:
    input_vote = st.slider("Vote Average", 0.0, 10.0, 6.0, 0.1)

if st.button("Predict"):
    genre_encoded = le.transform([input_genre])[0]
    X_new = pd.DataFrame([{
        "budget": input_budget,
        "popularity": input_popularity,
        "runtime": input_runtime,
        "vote_average": input_vote,
        "primary_genre": genre_encoded,
    }])[feature_cols]
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.success(f"🎉 Predicted: **Successful** (probability: {proba:.1%})")
    else:
        st.error(f"⚠️ Predicted: **Not Successful** (probability of success: {proba:.1%})")

st.caption("Model note: this synthetic dataset has weak feature-success correlations, "
           "so predictions lean toward the majority class (successful, ~81% of the data). "
           "Treat outputs as illustrative, not authoritative.")