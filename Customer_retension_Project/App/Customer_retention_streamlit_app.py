"""
Customer Engagement & Retention Dashboard
European Bank dataset (10,000 customers)

Modules
  1. Engagement vs churn overview
  2. Product utilization impact
  3. High-value disengaged customer detector
  4. Retention strength scoring (KPIs, Relationship Strength Index, sticky customers)
  5. Data quality checks
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Engagement & Retention", layout="wide")

# ── Constants ────────────────────────────────────────────────────────────────
CHURN = "#B5473A"
ACTIVE = "#2F6F8F"
INACTIVE = "#C58B2A"
STATUS_MAP = {"Active": ACTIVE, "Inactive": INACTIVE}
PROFILE_ORDER = [
    "Highly Engaged",
    "Active Low Product",
    "Inactive Low Engagement",
    "High Value Disengaged",
]
TIER_ORDER = ["Low", "Medium", "High"]

# The CSV is found relative to this file, so the app runs the same on a laptop
# and on Streamlit Community Cloud (no hardcoded C:\ paths).
HERE = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    HERE.parent / "DATA" / "European_Bank.csv",
    HERE.parent / "data" / "European_Bank.csv",
    HERE / "DATA" / "European_Bank.csv",
    HERE / "data" / "European_Bank.csv",
    HERE / "European_Bank.csv",
]


# ── Data loading and feature engineering ─────────────────────────────────────
@st.cache_data
def load_data():
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        return None, None, None

    raw = pd.read_csv(path)
    df = raw.drop(columns=["Surname", "Year"], errors="ignore").copy()

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min())

    df["Status"] = df["IsActiveMember"].map({1: "Active", 0: "Inactive"})

    # Relationship Strength Index (0-100):
    # 40% activity, 30% product count, 20% credit card, 10% tenure
    df["RSI"] = (
        100
        * (
            0.4 * df["IsActiveMember"]
            + 0.3 * minmax(df["NumOfProducts"])
            + 0.2 * df["HasCrCard"]
            + 0.1 * minmax(df["Tenure"])
        )
    ).round(1)

    # Tier cut-offs are fixed on the full dataset (tertiles), so they do not
    # move when the user applies filters.
    c1, c2 = df["RSI"].quantile([1 / 3, 2 / 3]).round(1).tolist()
    df["RSITier"] = pd.cut(
        df["RSI"], [-np.inf, c1, c2, np.inf], labels=TIER_ORDER
    ).astype(str)

    df["BalanceSalaryRatio"] = df["Balance"] / df["EstimatedSalary"]
    df["ProductGroup"] = pd.cut(
        df["NumOfProducts"], [0, 1, 2, 4], labels=["1 product", "2 products", "3-4 products"]
    ).astype(str)
    df["BalanceBand"] = pd.cut(
        df["Balance"],
        [-1, 0, 100000, 150000, np.inf],
        labels=["Zero", "Up to 100k", "100k-150k", "Above 150k"],
    ).astype(str)
    return raw, df, (c1, c2)


def add_profile(d, hv_threshold):
    """Engagement profile. Priority order resolves overlaps:
    Highly Engaged > High Value Disengaged > Active Low Product > Inactive Low Engagement."""
    conditions = [
        (d["IsActiveMember"] == 1) & (d["NumOfProducts"] >= 2),
        (d["IsActiveMember"] == 0) & (d["Balance"] > hv_threshold),
        (d["IsActiveMember"] == 1) & (d["NumOfProducts"] == 1),
    ]
    labels = ["Highly Engaged", "High Value Disengaged", "Active Low Product"]
    return np.select(conditions, labels, default="Inactive Low Engagement")


# ── Helpers ──────────────────────────────────────────────────────────────────
def pct(x, digits=1):
    return "n/a" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def ratio_text(x):
    return "n/a" if pd.isna(x) else f"{x:.2f}x"


def rate_table(d, by):
    t = d.groupby(by, observed=True)["Exited"].agg(Churn="mean", Customers="size").reset_index()
    t["Churn rate (%)"] = (t["Churn"] * 100).round(1)
    return t


def labelled(t, col, order=None):
    """Add an x-axis label that shows the group size, so small groups are visible."""
    t = t.copy()
    t[col] = t[col].astype(str)
    if order:
        t["_o"] = t[col].map({str(o): i for i, o in enumerate(order)})
        t = t.sort_values("_o").drop(columns="_o")
    t["Label"] = t[col] + "<br>n=" + t["Customers"].map("{:,}".format)
    return t


def churn_bar(t, title, xtitle=None, color=None, cmap=None):
    fig = px.bar(
        t,
        x="Label",
        y="Churn rate (%)",
        color=color,
        color_discrete_map=cmap,
        barmode="group",
        text="Churn rate (%)",
        hover_data=["Customers"],
        title=title,
        template="plotly_white",
    )
    if color is None:
        fig.update_traces(marker_color=CHURN)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="Churn rate (%)",
        yaxis_range=[0, max(t["Churn rate (%)"].max() * 1.25, 10)],
        xaxis_title=xtitle,
        legend_title_text=None,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    return fig


def compute_kpis(d, hv):
    act, ina = d[d["IsActiveMember"] == 1], d[d["IsActiveMember"] == 0]
    one, two = d[d["NumOfProducts"] == 1], d[d["NumOfProducts"] == 2]
    card, nocard = d[d["HasCrCard"] == 1], d[d["HasCrCard"] == 0]
    hb = d[d["Balance"] > hv]
    hb_inactive = hb[hb["IsActiveMember"] == 0]

    k = {}
    k["err"] = (
        ina["Exited"].mean() / act["Exited"].mean()
        if len(act) and len(ina) and act["Exited"].mean() > 0
        else np.nan
    )
    k["pdi"] = (
        (1 - two["Exited"].mean()) / (1 - one["Exited"].mean())
        if len(one) and len(two) and one["Exited"].mean() < 1
        else np.nan
    )
    k["hb_inactive_share"] = (hb["IsActiveMember"] == 0).mean() if len(hb) else np.nan
    k["hbdr"] = hb_inactive["Exited"].mean() if len(hb_inactive) else np.nan
    k["ccs"] = (
        ((1 - card["Exited"].mean()) - (1 - nocard["Exited"].mean())) * 100
        if len(card) and len(nocard)
        else np.nan
    )
    k["rsi"] = d["RSI"].mean()
    return k


# ── Load ─────────────────────────────────────────────────────────────────────
raw, df, rsi_cuts = load_data()
if raw is None:
    st.error(
        "European_Bank.csv was not found. Place it in a DATA folder next to the App folder "
        "(Customer_retension_Project/DATA/European_Bank.csv)."
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")
geography = st.sidebar.multiselect(
    "Geography", sorted(df["Geography"].unique()), default=sorted(df["Geography"].unique())
)
gender = st.sidebar.multiselect(
    "Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique())
)
status = st.sidebar.multiselect("Engagement status", ["Active", "Inactive"], default=["Active", "Inactive"])
products = st.sidebar.slider("Number of products", 1, 4, (1, 4))

bal_max = int(np.ceil(df["Balance"].max() / 10000) * 10000)
sal_max = int(np.ceil(df["EstimatedSalary"].max() / 10000) * 10000)
balance = st.sidebar.slider("Balance range", 0, bal_max, (0, bal_max), step=5000)
salary = st.sidebar.slider("Estimated salary range", 0, sal_max, (0, sal_max), step=5000)

st.sidebar.header("Detector thresholds")
hv_threshold = st.sidebar.slider(
    "High-value balance above", 50000, 200000, 100000, step=5000,
    help="Customers above this balance count as high-value. 100,000 matches the research paper.",
)
sal_threshold = st.sidebar.slider(
    "High salary at or above", 50000, sal_max, 150000, step=5000,
    help="Used to find high earners who keep almost nothing in the bank.",
)
ratio_threshold = st.sidebar.slider(
    "Balance-to-salary ratio at or above", 1.0, 10.0, 3.0, step=0.5,
    help="Flags inactive customers holding a balance far larger than their yearly salary.",
)

mask = (
    df["Geography"].isin(geography)
    & df["Gender"].isin(gender)
    & df["Status"].isin(status)
    & df["NumOfProducts"].between(*products)
    & df["Balance"].between(*balance)
    & df["EstimatedSalary"].between(*salary)
)
d = df[mask].copy()

st.title("Customer Engagement & Retention Dashboard")
st.caption(
    f"Showing {len(d):,} of {len(df):,} customers. Use the sidebar to filter and to set the detector thresholds."
)

if d.empty:
    st.warning("No customers match the current filters. Widen the filters in the sidebar.")
    st.stop()

d["EngagementProfile"] = add_profile(d, hv_threshold)
k = compute_kpis(d, hv_threshold)

tab_overview, tab_products, tab_detector, tab_strength, tab_quality = st.tabs(
    [
        "Engagement vs churn",
        "Product utilization",
        "High-value disengaged",
        "Retention strength",
        "Data quality",
    ]
)

# ── 1. Engagement vs churn ───────────────────────────────────────────────────
with tab_overview:
    c = st.columns(5)
    c[0].metric("Customers", f"{len(d):,}")
    c[1].metric("Churn rate", pct(d["Exited"].mean()))
    c[2].metric("Active members", pct(d["IsActiveMember"].mean()))
    c[3].metric(
        "Engagement Retention Ratio",
        ratio_text(k["err"]),
        help="Churn rate of inactive customers divided by churn rate of active customers.",
    )
    c[4].metric("Average products", f"{d['NumOfProducts'].mean():.2f}")

    left, right = st.columns(2)
    t = labelled(rate_table(d, "Status"), "Status", ["Active", "Inactive"])
    left.plotly_chart(churn_bar(t, "Churn rate: active vs inactive"))

    t = labelled(rate_table(d, "EngagementProfile"), "EngagementProfile", PROFILE_ORDER)
    right.plotly_chart(churn_bar(t, "Churn rate by engagement profile"))

    left, right = st.columns(2)
    t = labelled(rate_table(d, "Geography"), "Geography")
    left.plotly_chart(churn_bar(t, "Churn rate by country"))

    t = rate_table(d, ["Geography", "Status"])
    t["Label"] = t["Geography"]
    right.plotly_chart(
        churn_bar(t, "Churn rate by country and engagement status", color="Status", cmap=STATUS_MAP)
    )

    st.caption(
        "Profiles overlap on paper, so each customer gets one profile in this order: "
        "Highly Engaged, High Value Disengaged, Active Low Product, Inactive Low Engagement."
    )

# ── 2. Product utilization ───────────────────────────────────────────────────
with tab_products:
    tp = labelled(rate_table(d, "NumOfProducts"), "NumOfProducts")
    left, right = st.columns(2)
    left.plotly_chart(churn_bar(tp, "Churn rate by number of products", "Number of products"))

    tg = labelled(
        rate_table(d, "ProductGroup"), "ProductGroup", ["1 product", "2 products", "3-4 products"]
    )
    right.plotly_chart(churn_bar(tg, "Single vs multi-product customers", "Product group"))

    small = tp[tp["Customers"] < 100]
    if not small.empty:
        sizes = ", ".join(f"{r.NumOfProducts} products (n={r.Customers})" for r in small.itertuples())
        st.info(
            f"Small groups in the current view: {sizes}. Their churn rates are based on few "
            "customers, so treat them as indicative, not conclusive."
        )

    left, right = st.columns(2)
    t = rate_table(d, ["NumOfProducts", "Status"]).sort_values("NumOfProducts")
    t["Label"] = t["NumOfProducts"].astype(str)
    left.plotly_chart(
        churn_bar(
            t, "Churn rate by products and engagement status", "Number of products",
            color="Status", cmap=STATUS_MAP,
        )
    )

    t = rate_table(d, "HasCrCard")
    t["HasCrCard"] = t["HasCrCard"].map({1: "Has credit card", 0: "No credit card"})
    t = labelled(t, "HasCrCard", ["Has credit card", "No credit card"])
    right.plotly_chart(churn_bar(t, "Churn rate by credit card ownership"))

    st.metric(
        "Credit Card Stickiness Score",
        "n/a" if pd.isna(k["ccs"]) else f"{k['ccs']:+.1f} pp",
        help="Retention rate of card holders minus retention rate of non-holders, in percentage points. "
        "Close to zero means a card alone does not keep customers.",
    )

# ── 3. High-value disengaged detector ────────────────────────────────────────
with tab_detector:
    flags = pd.DataFrame(
        {
            f"High balance (above {hv_threshold:,}) and inactive": (d["Balance"] > hv_threshold)
            & (d["IsActiveMember"] == 0),
            f"Balance at least {ratio_threshold:g}x salary and inactive": (
                d["BalanceSalaryRatio"] >= ratio_threshold
            )
            & (d["IsActiveMember"] == 0),
            f"Salary at least {sal_threshold:,} but near-empty balance": (
                d["EstimatedSalary"] >= sal_threshold
            )
            & (d["BalanceSalaryRatio"] <= 0.05),
        }
    )
    flagged = flags.any(axis=1)
    d["Flags"] = flags.apply(lambda r: "; ".join(r.index[r]), axis=1)
    f = d[flagged]
    still = f[f["Exited"] == 0]

    c = st.columns(5)
    c[0].metric("Flagged customers", f"{len(f):,}")
    c[1].metric("Still with the bank", f"{len(still):,}")
    c[2].metric("Already churned", f"{int(f['Exited'].sum()):,}")
    c[3].metric("Churn rate of flagged", pct(f["Exited"].mean()) if len(f) else "n/a")
    c[4].metric(
        "Balance still at stake",
        f"{still['Balance'].sum() / 1e6:,.1f}M",
        help="Total balance held by flagged customers who have not left yet.",
    )

    summary = pd.DataFrame(
        {
            "Flag": flags.columns,
            "Customers": [int(flags[col].sum()) for col in flags.columns],
            "Still with the bank": [int((flags[col] & (d["Exited"] == 0)).sum()) for col in flags.columns],
            "Churn rate (%)": [
                round(d.loc[flags[col], "Exited"].mean() * 100, 1) if flags[col].any() else np.nan
                for col in flags.columns
            ],
        }
    )
    st.dataframe(summary, hide_index=True)

    if flagged.any():
        scatter = d.assign(Outcome=d["Exited"].map({1: "Churned", 0: "Retained"}), Flagged=flagged.map({True: "Flagged", False: "Not flagged"}))
        fig = px.scatter(
            scatter,
            x="EstimatedSalary",
            y="Balance",
            color="Flagged",
            color_discrete_map={"Flagged": CHURN, "Not flagged": "#C9CED3"},
            opacity=0.6,
            hover_data=["Outcome", "Status", "NumOfProducts"],
            title="Balance vs estimated salary",
            template="plotly_white",
        )
        fig.update_layout(legend_title_text=None, margin=dict(t=60, l=10, r=10, b=10))
        st.plotly_chart(fig)

    show_left = st.checkbox("Also show flagged customers who already left", value=False)
    view = f if show_left else still
    cols = [
        "CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance", "EstimatedSalary",
        "BalanceSalaryRatio", "NumOfProducts", "Status", "RSI", "Flags", "Exited",
    ]
    table = view[cols].sort_values("Balance", ascending=False).rename(
        columns={"Exited": "Churned (1=yes)", "BalanceSalaryRatio": "Balance/Salary"}
    )
    table["Balance/Salary"] = table["Balance/Salary"].round(2)
    st.dataframe(table, hide_index=True)

    contact = d[(flags.iloc[:, 0] | flags.iloc[:, 1]) & (d["Exited"] == 0)]
    st.download_button(
        f"Download inactive at-risk customers still with the bank ({len(contact):,}) as CSV",
        data=contact[cols].sort_values("Balance", ascending=False).to_csv(index=False).encode("utf-8"),
        file_name="at_risk_premium_customers.csv",
        mime="text/csv",
        disabled=contact.empty,
    )
    st.caption(
        "The download covers the two inactive-based flags only. High earners with a near-empty balance "
        "churn below the average, so treat that group as a cross-sell list, not an at-risk list."
    )

# ── 4. Retention strength ────────────────────────────────────────────────────
with tab_strength:
    c = st.columns(5)
    c[0].metric(
        "Engagement Retention Ratio", ratio_text(k["err"]),
        help="Churn rate (inactive) / churn rate (active).",
    )
    c[1].metric(
        "Product Depth Index", ratio_text(k["pdi"]),
        help="Retention rate of 2-product customers / retention rate of 1-product customers.",
    )
    c[2].metric(
        "High-Balance Disengagement Rate", pct(k["hbdr"]),
        help=f"Churn rate of inactive customers with balance above {hv_threshold:,}. "
        f"Share of high-balance customers who are inactive: {pct(k['hb_inactive_share'])}.",
    )
    c[3].metric(
        "Credit Card Stickiness Score",
        "n/a" if pd.isna(k["ccs"]) else f"{k['ccs']:+.1f} pp",
        help="Retention (card holders) minus retention (non-holders).",
    )
    c[4].metric(
        "Average RSI", f"{k['rsi']:.1f}",
        help="Relationship Strength Index, 0 to 100. See definitions below.",
    )

    left, right = st.columns(2)
    t = labelled(rate_table(d, "RSITier"), "RSITier", TIER_ORDER)
    left.plotly_chart(churn_bar(t, "Churn rate by Relationship Strength tier", "RSI tier"))

    bins = list(range(0, 101, 10))
    band_labels = [f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])]
    d["RSIBand"] = pd.cut(d["RSI"], bins=bins, labels=band_labels, include_lowest=True)
    band = rate_table(d, "RSIBand")
    band = band[band["Customers"] >= 30]
    fig = px.line(
        band, x="RSIBand", y="Churn rate (%)", markers=True,
        hover_data=["Customers"], title="Churn rate by RSI band (engagement threshold view)",
        template="plotly_white",
    )
    fig.update_traces(line_color=CHURN, marker_color=CHURN)
    fig.update_layout(
        xaxis_title="RSI band", yaxis_title="Churn rate (%)", margin=dict(t=60, l=10, r=10, b=10)
    )
    right.plotly_chart(fig)
    st.caption(
        f"Tiers use fixed tertile cut-offs from the full dataset: Low is RSI up to {rsi_cuts[0]:g}, "
        f"Medium is above {rsi_cuts[0]:g} up to {rsi_cuts[1]:g}, High is above {rsi_cuts[1]:g}. "
        "RSI bands with fewer than 30 customers are hidden."
    )

    high = d[d["RSITier"] == "High"]
    sticky = high[high["NumOfProducts"] <= 2]
    excluded = len(high) - len(sticky)
    low = d[d["RSITier"] == "Low"]
    c = st.columns(3)
    c[0].metric(
        "Sticky customers (High tier, up to 2 products)", f"{len(sticky):,}",
        delta=f"{len(sticky) / len(d) * 100:.1f}% of view", delta_color="off",
        help="High RSI tier, excluding customers with 3 or 4 products, who churn heavily.",
    )
    c[1].metric("Sticky churn rate", pct(sticky["Exited"].mean()) if len(sticky) else "n/a")
    c[2].metric("Low-tier churn rate", pct(low["Exited"].mean()) if len(low) else "n/a")
    if excluded:
        st.caption(
            f"{excluded:,} High-tier customers with 3 or 4 products are excluded from the sticky group "
            f"(their churn rate is {pct(high[high['NumOfProducts'] >= 3]['Exited'].mean())})."
        )

    t = rate_table(d, ["RSITier", "Geography"])
    t["_o"] = t["RSITier"].map({o: i for i, o in enumerate(TIER_ORDER)})
    t = t.sort_values("_o")
    t["Label"] = t["RSITier"]
    st.plotly_chart(
        churn_bar(t, "Churn stability: RSI tier within each country", "RSI tier", color="Geography")
    )

# ── 5. Data quality ──────────────────────────────────────────────────────────
with tab_quality:
    def check(ok):
        return "Pass" if ok else "Fail"

    rows = [
        ("Rows x columns", f"{raw.shape[0]:,} x {raw.shape[1]}", "Info"),
        ("Missing values", str(int(raw.isna().sum().sum())), check(raw.isna().sum().sum() == 0)),
        ("Duplicate CustomerId", str(int(raw["CustomerId"].duplicated().sum())), check(not raw["CustomerId"].duplicated().any())),
    ]
    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        bad = int((~raw[col].isin([0, 1])).sum())
        rows.append((f"{col} contains only 0/1", f"{bad} invalid", check(bad == 0)))
    rows += [
        ("NumOfProducts within 1-4", f"{raw['NumOfProducts'].min()} to {raw['NumOfProducts'].max()}", check(raw["NumOfProducts"].between(1, 4).all())),
        ("Balance is not negative", f"min {raw['Balance'].min():,.2f}", check((raw["Balance"] >= 0).all())),
        ("Age is positive", f"min {raw['Age'].min()}", check((raw["Age"] > 0).all())),
        ("Countries present", ", ".join(sorted(raw["Geography"].unique())), "Info"),
        ("Overall churn rate (Exited.mean())", f"{raw['Exited'].mean():.2%}", "Info"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Check", "Result", "Status"]), hide_index=True)

# ── Definitions ──────────────────────────────────────────────────────────────
with st.expander("How the profiles and indices are defined"):
    st.markdown(
        """
**Engagement profiles** (assigned in this order)
- Highly Engaged: active and 2 or more products
- High Value Disengaged: inactive and balance above the high-value threshold
- Active Low Product: active with 1 product
- Inactive Low Engagement: everyone else

**Relationship Strength Index (RSI, 0 to 100)**
`100 x (0.4 x Active + 0.3 x scaled products + 0.2 x HasCrCard + 0.1 x scaled tenure)`.
Products and tenure are min-max scaled. The tiers split the full dataset into thirds.
Note that RSI scores 3 and 4 products higher, yet those customers churn heavily, so check the
product charts alongside the RSI view.

**Sticky customer:** a customer in the High RSI tier with at most 2 products. Customers with 3 or 4 products are excluded because they churn heavily.

**KPIs**
- Engagement Retention Ratio: churn (inactive) / churn (active)
- Product Depth Index: retention (2 products) / retention (1 product)
- High-Balance Disengagement Rate: churn rate of inactive customers above the balance threshold
- Credit Card Stickiness Score: retention (card) minus retention (no card), in percentage points
"""
    )
