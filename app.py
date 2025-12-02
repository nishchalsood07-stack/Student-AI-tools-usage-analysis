from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------
# 🛠️  Page configuration & Altair settings
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Student AI‑Tool Survey Explorer",
    page_icon="🧠",
    layout="wide",
)
alt.themes.enable("default")
alt.data_transformers.disable_max_rows()

# ---------------------------------------------------------------------
# 📥  Data loading (cached)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Loading survey data …")
def load_data(path: str = "Students.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert all relevant columns to string to avoid type issues
    for col in [
        "State", "Device_Used", "Preferred_AI_Tool",
        "AI_Tools_Used", "Trust_in_AI_Tools", "Year_of_Study"
    ]:
        df[col] = df[col].astype(str)
    return df

survey_df = load_data()

# ---------------------------------------------------------------------
# 🔍  Sidebar filters
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filter responses")

    tools_selected = st.multiselect(
        "Preferred AI Tool(s)",
        options=survey_df["Preferred_AI_Tool"].dropna().unique().tolist(),
        default=survey_df["Preferred_AI_Tool"].dropna().unique().tolist(),
    )

    devices_selected = st.multiselect(
        "Device Used", survey_df["Device_Used"].dropna().unique().tolist(),
        default=survey_df["Device_Used"].dropna().unique().tolist(),
    )

    states_selected = st.multiselect(
        "State", survey_df["State"].dropna().unique().tolist(),
        default=survey_df["State"].dropna().unique().tolist(),
    )

    years_selected = st.multiselect(
        "Year of Study", survey_df["Year_of_Study"].dropna().unique().tolist(),
        default=survey_df["Year_of_Study"].dropna().unique().tolist(),
    )

    st.markdown("---")
    st.caption("Rows after filtering appear in the header metrics →")

filtered = survey_df[
    survey_df["Preferred_AI_Tool"].isin(tools_selected)
    & survey_df["Device_Used"].isin(devices_selected)
    & survey_df["State"].isin(states_selected)
    & survey_df["Year_of_Study"].isin(years_selected)
]

# ---------------------------------------------------------------------
# 🖥️  Header metrics and raw‑data viewer
# ---------------------------------------------------------------------
st.title("🧠 Student AI‑Tool Survey Explorer")

col1, col2 = st.columns(2)
col1.metric("Rows (filtered)", f"{len(filtered):,}")
col2.metric("Columns", f"{filtered.shape[1]}")

with st.expander("📋 Show raw data"):
    st.dataframe(filtered, use_container_width=True)

# ---------------------------------------------------------------------
# 📊  Tabbed analytics section
# ---------------------------------------------------------------------
usage_tab, heat_tab, trust_tab, awareness_tab, predict_tab = st.tabs(
    [
        "Usage by Tool & Device",
        "State‑wise Usage",
        "Trust vs Year",
        "Awareness vs Tools",
        "🔮 Prediction",
    ]
)

# 1️⃣ Usage by Tool & Device
with usage_tab:
    st.subheader("Usage by AI Tool & Device")

    usage_agg = (
        filtered.groupby(["Preferred_AI_Tool", "Device_Used"])["Daily_Usage_Hours"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "Total_Usage_Hours", "count": "Response_Count"})
    )

    usage_pivot = (
        usage_agg.pivot(index="Preferred_AI_Tool", columns="Device_Used", values="Total_Usage_Hours")
        .fillna(0)
        .round(1)
    )
    count_pivot = (
        usage_agg.pivot(index="Preferred_AI_Tool", columns="Device_Used", values="Response_Count")
        .fillna(0)
        .astype(int)
    )

    col_a, col_b = st.columns(2)
    col_a.metric("Unique tools", len(usage_pivot))
    col_b.metric("Unique devices", len(usage_pivot.columns))

    st.dataframe(usage_pivot, use_container_width=True)

    usage_totals = usage_pivot.sum(axis=1).reset_index()
    usage_totals.columns = ["Tool", "Total_Hours"]

    chart1 = (
        alt.Chart(usage_totals)
        .mark_bar()
        .encode(
            y=alt.Y("Tool:N", sort="-x", title="Preferred AI Tool"),
            x=alt.X("Total_Hours:Q", title="Total Daily Usage Hours"),
            tooltip=["Tool", "Total_Hours"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart1, use_container_width=True)

    usage_long = usage_pivot.reset_index().melt(
        id_vars="Preferred_AI_Tool", var_name="Device", value_name="Hours"
    )
    chart2 = (
        alt.Chart(usage_long)
        .mark_bar()
        .encode(
            x=alt.X("Preferred_AI_Tool:N", title="Preferred AI Tool"),
            y=alt.Y("Hours:Q", stack="normalize", title="Share of Daily Hours"),
            color="Device:N",
            tooltip=["Device", "Hours"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart2, use_container_width=True)

# 2️⃣ Heatmap – Usage by State & Tool
with heat_tab:
    st.subheader("Usage Hours by State & Tool")

    heatmap_df = (
        filtered.groupby(["Preferred_AI_Tool", "State"])["Daily_Usage_Hours"]
        .sum()
        .reset_index()
    )

    chart3 = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("State:N", title="State"),
            y=alt.Y("Preferred_AI_Tool:N", title="Preferred AI Tool"),
            color=alt.Color("Daily_Usage_Hours:Q", scale=alt.Scale(scheme="greens")),
            tooltip=["Preferred_AI_Tool", "State", "Daily_Usage_Hours"],
        )
        .properties(height=500)
    )
    st.altair_chart(chart3, use_container_width=True)

# 3️⃣ Trust in AI Tools vs Year
with trust_tab:
    st.subheader("Trust in AI Tools vs Year of Study")

    trust_counts = (
        filtered.groupby(["Year_of_Study", "Trust_in_AI_Tools"])
        .size()
        .reset_index(name="Count")
    )

    chart4 = (
        alt.Chart(trust_counts)
        .mark_bar()
        .encode(
            x=alt.X("Trust_in_AI_Tools:N", title="Trust Level"),
            y=alt.Y("Count:Q", title="Number of Responses"),
            color="Year_of_Study:N",
            tooltip=["Year_of_Study", "Count"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart4, use_container_width=True)

# 4️⃣ Awareness Level vs AI Tools Used
with awareness_tab:
    st.subheader("Awareness Level vs AI Tools Used")

    awareness_counts = (
        filtered.groupby(["Awareness_Level", "AI_Tools_Used"])
        .size()
        .reset_index(name="Count")
    )

    chart5 = (
        alt.Chart(awareness_counts)
        .mark_bar()
        .encode(
            x=alt.X("Awareness_Level:N", title="Awareness Level"),
            y=alt.Y("Count:Q", title="Number of Responses"),
            color="AI_Tools_Used:N",
            tooltip=["AI_Tools_Used", "Count"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart5, use_container_width=True)

# 5️⃣ 🔮 Prediction Tab
with predict_tab:
    st.subheader("🔮 Predict a Student's Daily Usage Hours")

    feature_cols = [
        "Year_of_Study",
        "AI_Tools_Used",
        "Trust_in_AI_Tools",
        "Preferred_AI_Tool",
        "Device_Used",
        "State",
    ]

    @st.cache_resource(show_spinner="Training model …")
    def train_model(df: pd.DataFrame):
        X = pd.get_dummies(df[feature_cols], drop_first=True)
        y = df["Daily_Usage_Hours"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return model, X.columns, mae, rmse

    model, dummy_cols, mae, rmse = train_model(survey_df)

    st.write(f"**Model validation** — MAE: `{mae:.2f}` hrs   RMSE: `{rmse:.2f}` hrs")
    st.markdown("---")
    st.markdown("#### Try it yourself")

    # 5️⃣ 🔮 Prediction – Estimate Daily Usage Hours
with predict_tab:
    st.subheader("🔮 Predict a Student's Daily Usage Hours")

    feature_cols = [
        "Year_of_Study",
        "AI_Tools_Used",
        "Trust_in_AI_Tools",
        "Preferred_AI_Tool",
        "Device_Used",
        "State",
    ]

    # Train model on full dataset
    @st.cache_resource(show_spinner="Training model...")
    def train_model_full(df):
        X = pd.get_dummies(df[feature_cols])  # No drop_first
        y = df["Daily_Usage_Hours"]
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        return model, X.columns.tolist()

    model, model_columns = train_model_full(survey_df)

    st.markdown("#### Try it yourself")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)

        year = c1.selectbox("Year of Study", sorted(survey_df["Year_of_Study"].unique()))
        ai_tools_used = c1.selectbox("AI Tools Used", sorted(survey_df["AI_Tools_Used"].unique()))
        trust = c1.selectbox("Trust in AI Tools", sorted(survey_df["Trust_in_AI_Tools"].unique()))

        preferred = c2.selectbox("Preferred AI Tool", sorted(survey_df["Preferred_AI_Tool"].unique()))
        device = c2.selectbox("Device Used", sorted(survey_df["Device_Used"].unique()))
        state = c2.selectbox("State", sorted(survey_df["State"].unique()))

        submitted = st.form_submit_button("Predict")

    if submitted:
        user_input = pd.DataFrame([{
            "Year_of_Study": year,
            "AI_Tools_Used": ai_tools_used,
            "Trust_in_AI_Tools": trust,
            "Preferred_AI_Tool": preferred,
            "Device_Used": device,
            "State": state,
        }])

        # Encode input using same dummy logic
        user_encoded = pd.get_dummies(user_input)

        # Add any missing columns from model
        for col in model_columns:
            if col not in user_encoded.columns:
                user_encoded[col] = 0

        # Ensure correct column order
        user_encoded = user_encoded[model_columns]

        # Predict
        prediction = model.predict(user_encoded)[0]
        st.success(f"Estimated daily usage for this profile: **{prediction:.2f} hours**")
