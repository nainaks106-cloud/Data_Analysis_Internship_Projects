import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ─── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title="Bank Retention Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ─── LOAD DATA ─────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\NAINA\OneDrive\Documents\GitHub\Data_Analysis_Internship_Projects\Data_Analysis_Internship_Projects\Customer_retension_Project\DATA\European_Bank.csv")
    df.drop(['CustomerId', 'Surname', 'Year'], axis=1, inplace=True)

    def engagement_profile(row):
        if row['IsActiveMember'] == 1 and row['NumOfProducts'] >= 2:
            return 'Highly Engaged'
        elif row['IsActiveMember'] == 0 and row['Balance'] > 100000:
            return 'High Value Disengaged'
        elif row['IsActiveMember'] == 1 and row['NumOfProducts'] == 1:
            return 'Active Low Product'
        else:
            return 'Inactive Low Engagement'

    df['EngagementProfile'] = df.apply(engagement_profile, axis=1)
    df['RelationshipScore'] = (
        df['IsActiveMember'] * 40 +
        df['NumOfProducts'] * 20 +
        df['HasCrCard'] * 10
    )
    return df

df = load_data()

# ─── TITLE ─────────────────────────────────────
st.title("🏦 Bank Customer Retention Dashboard")
st.markdown("**European Central Bank — Customer Engagement Analytics**")
st.markdown("---")

# ─── SIDEBAR FILTERS ───────────────────────────
st.sidebar.header("🔍 Filters")

geography = st.sidebar.multiselect(
    "Select Geography",
    options=df['Geography'].unique(),
    default=df['Geography'].unique()
)

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

active = st.sidebar.multiselect(
    "Active Member",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "Active" if x == 1 else "Inactive"
)

products = st.sidebar.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=(1, 4)
)

balance = st.sidebar.slider(
    "Balance Range",
    min_value=0,
    max_value=250000,
    value=(0, 250000),
    step=10000
)

# ─── APPLY FILTERS ─────────────────────────────
filtered = df[
    (df['Geography'].isin(geography)) &
    (df['Gender'].isin(gender)) &
    (df['IsActiveMember'].isin(active)) &
    (df['NumOfProducts'].between(products[0], products[1])) &
    (df['Balance'].between(balance[0], balance[1]))
]

# ─── KPI CARDS ─────────────────────────────────
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(filtered))

with col2:
    churn_rate = round(filtered['Exited'].mean() * 100, 2)
    st.metric("Churn Rate", f"{churn_rate}%")

with col3:
    active_pct = round(filtered['IsActiveMember'].mean() * 100, 2)
    st.metric("Active Members", f"{active_pct}%")

with col4:
    avg_products = round(filtered['NumOfProducts'].mean(), 2)
    st.metric("Avg Products", avg_products)

st.markdown("---")

# ─── ROW 1 CHARTS ──────────────────────────────
st.subheader("📈 Engagement & Product Analysis")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        filtered, x='IsActiveMember',
        color='Exited',
        barmode='group',
        title='Active vs Inactive Member Churn',
        labels={'IsActiveMember': 'Is Active (1=Yes, 0=No)'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        filtered, x='NumOfProducts',
        color='Exited',
        barmode='group',
        title='Number of Products vs Churn',
        labels={'NumOfProducts': 'Number of Products'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── ROW 2 CHARTS ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        filtered, x='Geography',
        color='Exited',
        barmode='group',
        title='Geography vs Churn'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        filtered, x='EngagementProfile',
        color='Exited',
        barmode='group',
        title='Engagement Profile vs Churn'
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── ROW 3 CHARTS ──────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        filtered, x='Exited', y='Age',
        title='Age vs Churn',
        labels={'Exited': 'Exited (0=Stayed, 1=Churned)'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        filtered, x='Exited', y='Balance',
        title='Balance vs Churn',
        labels={'Exited': 'Exited (0=Stayed, 1=Churned)'}
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─── PREMIUM AT RISK TABLE ─────────────────────
st.subheader("🚨 High Value Disengaged Customers")

risk_df = filtered[
    (filtered['Balance'] > 100000) &
    (filtered['IsActiveMember'] == 0)
][['Age', 'Geography', 'Gender', 'Balance',
   'NumOfProducts', 'RelationshipScore', 'Exited']]

st.write(f"Total At-Risk Premium Customers: **{len(risk_df)}**")
st.dataframe(risk_df, use_container_width=True)

st.markdown("---")

# ─── RETENTION STRENGTH PANEL ──────────────────
st.subheader("💪 Retention Strength by Engagement Profile")

retention = filtered.groupby('EngagementProfile')['Exited'].mean().reset_index()
retention.columns = ['EngagementProfile', 'ChurnRate']
retention['ChurnRate'] = (retention['ChurnRate'] * 100).round(2)
retention['RetentionRate'] = 100 - retention['ChurnRate']

fig = px.bar(
    retention,
    x='EngagementProfile',
    y='RetentionRate',
    title='Retention Rate by Engagement Profile',
    color='RetentionRate',
    color_continuous_scale='Greens'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Dashboard built for European Central Bank — Customer Retention Strategy Project")