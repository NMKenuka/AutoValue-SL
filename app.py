"""AutoValue SL — Ensemble-learning dashboard.

Run: python -m streamlit run app.py
Required next to this file: car_price_dataset.csv, stacking_model.pkl,
preprocessor.pkl, feature_columns.pkl
Optional: test_predictions.csv (created in Jupyter for true holdout charts).
"""

from pathlib import Path
from html import escape

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AutoValue SL | Vehicle Price Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "car_price_dataset.csv"
MODEL_FILE = ROOT / "stacking_model.pkl"
PREPROCESSOR_FILE = ROOT / "preprocessor.pkl"
COLUMNS_FILE = ROOT / "feature_columns.pkl"
TEST_PREDICTIONS_FILE = ROOT / "test_predictions.csv"

# Results measured in the user's Jupyter Notebook on the held-out test set.
RESULTS = pd.DataFrame(
    [
        {"Model": "Bagging", "Algorithm": "Random Forest", "MAE": 5.5183, "RMSE": 21.6465, "R²": 0.8027},
        {"Model": "Boosting", "Algorithm": "Gradient Boosting", "MAE": 9.1663, "RMSE": 23.9049, "R²": 0.7594},
        {"Model": "Stacking", "Algorithm": "RF + GB + KNN → Ridge", "MAE": 5.7400, "RMSE": 21.4913, "R²": 0.8055},
    ]
)

NAVY = "#0B1220"
PANEL = "#121F32"
TEAL = "#39D6C5"
BLUE = "#68A7FF"
AMBER = "#F8BD70"
TEXT = "#EDF4FF"
MUTED = "#A3B6D0"
MODEL_COLORS = {"Bagging": BLUE, "Boosting": AMBER, "Stacking": TEAL}

st.markdown(
    """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stApp"] { font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
[data-testid="stAppViewContainer"] { background: #0B1220; color: #EDF4FF; }
[data-testid="stHeader"] { background: rgba(11,18,32,.82); }
[data-testid="stSidebar"] { background: #101C2D; border-right: 1px solid #29384F; }
[data-testid="stSidebar"] * { color: #EDF4FF; }
.block-container { max-width: 1320px; padding: 2.2rem 2.5rem 4rem; }
header[data-testid="stHeader"] { color: #EDF4FF; }
h1,h2,h3,h4, p, label { color: #EDF4FF; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: #D4E2F4 !important; font-weight: 600; }
[data-testid="stMarkdownContainer"] p { line-height: 1.55; }
[data-testid="stTabs"] [role="tab"] { color: #C8D6E9 !important; opacity: 1 !important; font-weight: 650; }
[data-testid="stTabs"] [role="tab"] *, [data-testid="stTabs"] [role="tab"] p { color: inherit !important; opacity: 1 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #39D6C5 !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #39D6C5 !important; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stNumberInput"] input { background: #18283E !important; color: #F4F8FF !important; }
[data-baseweb="select"] span, [data-baseweb="input"] input { color: #F4F8FF !important; }
[data-testid="stSelectbox"] div, [data-testid="stNumberInput"] div { border-color: #3B4E66; }
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] { background: #39D6C5; color: #071824 !important; border: none; border-radius: 11px; font-weight: 800; }
[data-testid="stFormSubmitButton"] button[kind="primary"] p { color: #071824 !important; }
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover { background: #70E9DD; }
[data-testid="stMetric"] { background: #142239; border: 1px solid #293C54; padding: 1.05rem; border-radius: 16px; }
[data-testid="stMetricLabel"] p { color: #A3B6D0 !important; }
[data-testid="stMetricValue"] { color: #EDF4FF !important; }
.hero { background: linear-gradient(115deg,#152A46 0%,#12364B 58%,#0B4A4D 100%); border: 1px solid #315167; border-radius: 23px; padding: 2rem 2.1rem; margin-bottom: 1.3rem; }
.eyebrow { color:#85E6DF; font-size:.78rem; text-transform:uppercase; font-weight:800; letter-spacing:.13em; }
.hero h1 { color:#FFFFFF !important; margin:.55rem 0 .65rem; font-weight:800; font-size:clamp(2rem,3.3vw,3.05rem); letter-spacing:-.04em; }
.hero p { color:#CFDFEF !important; margin:0; max-width:800px; line-height:1.65; }
.kpi { background:#142239; border:1px solid #31415B; border-radius:17px; padding:1.2rem 1.3rem; min-height:138px; }
.kpi-label { color:#9EB2CE; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; font-weight:750; }
.kpi-value { color:#FFFFFF; font-size:1.95rem; font-weight:800; line-height:1.5; }
.kpi-note { color:#A3B6D0; font-size:.82rem; }
.caption-muted { color:#A3B6D0 !important; font-size:.89rem; }
.result-box { padding:1.5rem 1.8rem; border-radius:18px; background:linear-gradient(120deg,#153F45,#123C55); border:1px solid #298D91; margin:1rem 0; }
.result-box .result-label { color:#BFEFEA; text-transform:uppercase; font-weight:800; font-size:.8rem; letter-spacing:.08em; }
.result-box .result-price { color:white; font-weight:850; font-size:clamp(2.5rem,5vw,4rem); letter-spacing:-.04em; line-height:1.25; }
.result-box .result-note { color:#C5E9E5; font-size:.86rem; }
.score-table { width:100%; border-collapse:collapse; font-size:.94rem; overflow:hidden; }
.score-table th { text-align:left; color:#B4C6DE; background:#17283D; padding:12px; }
.score-table td { padding:13px 12px; border-bottom:1px solid #304158; color:#E8F0FC; }
.score-table tr:last-child td { border-bottom:none; }
.score-table .best { color:#72EBCE; font-weight:800; }
.steps { border-left:2px solid #39D6C5; padding-left:1.1rem; margin:1rem 0; }
.steps p { color:#CEDBEC !important; }
.small-muted { color:#A3B6D0; font-size:.84rem; }
@media (max-width: 760px) { .block-container { padding:1.3rem 1rem 3rem; } .hero {padding:1.35rem;} .kpi-value {font-size:1.55rem;} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_FILE)


@st.cache_resource(show_spinner="Loading your trained Stacking model…")
def load_model():
    # Only load model artifacts created in your own trusted Jupyter Notebook.
    return (
        joblib.load(MODEL_FILE),
        joblib.load(PREPROCESSOR_FILE),
        list(joblib.load(COLUMNS_FILE)),
    )


def clean_options(frame, col):
    return sorted(frame[col].dropna().astype(str).unique().tolist())


def default_number(frame, col, fallback, kind=float):
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    value = float(values.median()) if not values.empty else float(fallback)
    return int(round(value)) if kind is int else float(value)


def kpi(label, value, note):
    st.markdown(
        '<div class="kpi"><div class="kpi-label">' + escape(label) +
        '</div><div class="kpi-value">' + escape(str(value)) +
        '</div><div class="kpi-note">' + escape(note) + '</div></div>',
        unsafe_allow_html=True,
    )


def style_plot(fig, ytitle=""):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT),
        height=355,
        margin=dict(l=16, r=16, t=35, b=25),
        yaxis_title=ytitle,
        legend=dict(orientation="h", y=1.17, x=0),
        hoverlabel=dict(bgcolor="#1B3047", font_color="#FFFFFF"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#35455D")
    fig.update_yaxes(gridcolor="#2B3D52", zerolinecolor="#35455D")
    return fig


if not DATA_FILE.exists():
    st.error("Missing car_price_dataset.csv. Place it alongside app.py.")
    st.stop()

try:
    raw = load_data()
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

# Sidebar inputs are independent of selected tabs; users can adjust them anytime.
with st.sidebar:
    st.markdown("##Vehicle inputs")
    st.caption("Choose your vehicle's details and select **Estimate price**.")

    brands = clean_options(raw, "Brand")
    brand_index = brands.index("TOYOTA") if "TOYOTA" in brands else 0
    brand = st.selectbox("Brand", brands, index=brand_index, key="brand")
    brand_data = raw.loc[raw["Brand"].astype(str) == brand]
    models = clean_options(brand_data, "Model")

    with st.form("vehicle_inputs"):
        model_name = st.selectbox("Model", models)
        selected_rows = brand_data.loc[brand_data["Model"].astype(str) == model_name]
        typical = selected_rows if len(selected_rows) else brand_data

        st.markdown("#### Specifications")
        yom = st.number_input(
            "Manufacturing year", min_value=1950, max_value=2026,
            value=max(1950, min(2026, default_number(typical, "YOM", 2018, int))), step=1,
        )
        engine = st.number_input(
            "Engine capacity (cc)", min_value=0.0,
            value=max(0.0, default_number(typical, "Engine (cc)", 1500)), step=100.0,
        )
        mileage = st.number_input(
            "Mileage (km)", min_value=0.0,
            value=max(0.0, default_number(typical, "Millage(KM)", 50000)), step=1000.0,
        )
        gear = st.selectbox("Transmission", clean_options(raw, "Gear"))
        fuel = st.selectbox("Fuel type", clean_options(raw, "Fuel Type"))

        st.markdown("#### Listing details")
        towns = clean_options(raw, "Town")
        town = st.selectbox("Town", towns, index=towns.index("Colombo") if "Colombo" in towns else 0)
        condition = st.selectbox("Condition", clean_options(raw, "Condition"))
        leasing = st.selectbox("Leasing", clean_options(raw, "Leasing"))

        st.markdown("#### Equipment")
        equipment = {}
        for col, label in [
            ("AIR CONDITION", "Air conditioning"),
            ("POWER STEERING", "Power steering"),
            ("POWER MIRROR", "Power mirrors"),
            ("POWER WINDOW", "Power windows"),
        ]:
            choices = clean_options(raw, col)
            equipment[col] = st.selectbox(label, choices, index=choices.index("Available") if "Available" in choices else 0)

        submit = st.form_submit_button("Estimate price", type="primary", use_container_width=True)

    st.caption("Historical listings · educational estimation, not a guaranteed sale price.")

    if submit:
        user_input = {
            "Brand": brand,
            "Model": model_name,
            "YOM": int(yom),
            "Engine (cc)": float(engine),
            "Gear": gear,
            "Fuel Type": fuel,
            "Millage(KM)": float(mileage),
            "Town": town,
            "Leasing": leasing,
            "Condition": condition,
            **equipment,
        }
        missing_artifacts = [p.name for p in (MODEL_FILE, PREPROCESSOR_FILE, COLUMNS_FILE) if not p.exists()]
        if missing_artifacts:
            st.session_state["prediction_error"] = "Missing model file(s): " + ", ".join(missing_artifacts)
            st.session_state.pop("prediction", None)
        else:
            try:
                model, transformer, columns = load_model()
                absent = [col for col in columns if col not in user_input]
                if absent:
                    raise ValueError("The model requires additional features: " + ", ".join(absent))
                input_df = pd.DataFrame([user_input], columns=columns)
                transformed = transformer.transform(input_df)
                predicted = float(model.predict(transformed)[0])
                if not np.isfinite(predicted):
                    raise ValueError("Model returned a non-finite value.")
                st.session_state["prediction"] = (predicted, f"{brand} {model_name} · {yom}")
                st.session_state.pop("prediction_error", None)
            except Exception as exc:
                st.session_state["prediction_error"] = f"Could not run the prediction: {exc}"
                st.session_state.pop("prediction", None)

st.markdown(
    """<div class="hero"><div class="eyebrow">AutoValue SL / Ensemble Learning Project</div>
    <h1>Know the data.<br>Estimate the value.</h1>
    <p>Explore how Bagging, Boosting and Stacking perform on historical Sri Lankan vehicle listings, then get a personalized estimate from your trained Stacking ensemble.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    kpi("Lowest test RMSE", "21.4913", "Stacking · lower is better")
with c2:
    kpi("Lowest test MAE", "5.5183", "Random Forest · lower is better")
with c3:
    kpi("Highest test R²", "0.8055", "Stacking · higher is better")

st.write("")
predict_tab, comparison_tab, insights_tab, about_tab = st.tabs(
    ["Prediction", " Model comparison", " Data insights", " About the project"]
)

with predict_tab:
    st.subheader("Your vehicle price estimate")
    st.markdown("<p class='caption-muted'>Use the form in the left sidebar. Your estimate appears here without retraining the model.</p>", unsafe_allow_html=True)
    if "prediction_error" in st.session_state:
        st.error(st.session_state["prediction_error"])
        st.info("The comparison charts and dataset insights still work without a loaded model. Verify the saved .pkl files and package versions if prediction fails.")
    if "prediction" in st.session_state:
        value, vehicle = st.session_state["prediction"]
        st.markdown(
            '<div class="result-box"><div class="result-label">Predicted price · dataset units</div>'
            f'<div class="result-price">{value:,.2f}</div>'
            f'<div class="result-note">{escape(vehicle)} · Stacking Regressor</div></div>',
            unsafe_allow_html=True,
        )
        st.warning("Price units have not been independently verified. Do not interpret the number as LKR until the dataset's units are confirmed.")
    else:
        st.info("Complete the vehicle details in the sidebar and click **Estimate price**.")
    st.markdown("#### How the estimate is produced")
    st.markdown("**Vehicle inputs** → **One-hot encoding + numerical scaling** → **Stacking Regressor** → **Estimated price**")
    st.caption("The model uses Random Forest, Gradient Boosting and KNN as base learners, with Ridge as the meta-model.")

with comparison_tab:
    st.subheader("Compare three ensemble approaches")
    st.caption("These are your real Jupyter Notebook results from the same held-out test set. No models are retrained in the dashboard.")
    ranked = RESULTS.sort_values("RMSE").reset_index(drop=True)
    rows = []
    for _, item in ranked.iterrows():
        name = escape(str(item["Model"]))
        rmse_class = ' class="best"' if item["Model"] == "Stacking" else ""
        mae_class = ' class="best"' if item["Model"] == "Bagging" else ""
        rows.append(
            f"<tr><td><b>{name}</b><br><small>{escape(str(item['Algorithm']))}</small></td>"
            f"<td{mae_class}>{item['MAE']:.4f}</td><td{rmse_class}>{item['RMSE']:.4f}</td>"
            f"<td>{item['R²']:.4f}</td></tr>"
        )
    st.markdown(
        '<table class="score-table"><thead><tr><th>Approach</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th></tr></thead>'
        '<tbody>' + "".join(rows) + '</tbody></table>', unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Prediction errors")
        error_fig = go.Figure()
        for metric, color in [("MAE", BLUE), ("RMSE", TEAL)]:
            error_fig.add_trace(go.Bar(
                name=metric, x=RESULTS["Model"], y=RESULTS[metric],
                marker_color=color, text=[f"{v:.2f}" for v in RESULTS[metric]],
                textposition="outside", hovertemplate=f"%{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>",
            ))
        error_fig.update_layout(barmode="group", yaxis=dict(range=[0, 29]))
        st.plotly_chart(style_plot(error_fig, "Error · dataset price units"), use_container_width=True)
    with right:
        st.markdown("#### Explained variance (R²)")
        r2_fig = go.Figure(go.Bar(
            x=RESULTS["Model"], y=RESULTS["R²"],
            marker_color=[MODEL_COLORS[m] for m in RESULTS["Model"]],
            text=[f"{v:.4f}" for v in RESULTS["R²"]], textposition="outside",
            hovertemplate="%{x}<br>R²: %{y:.4f}<extra></extra>",
        ))
        r2_fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(style_plot(r2_fig, "R² score"), use_container_width=True)
    st.info("**Interpretation:** Stacking has the lowest RMSE and highest R², while Random Forest has the lowest MAE. Stacking's RMSE advantage over Random Forest is only about 0.72%, so it is a slight—not definitive—improvement.")
    st.download_button("Download comparison CSV", RESULTS.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")

    st.markdown("#### Actual vs. predicted prices (held-out test data)")
    if TEST_PREDICTIONS_FILE.exists():
        try:
            preds = pd.read_csv(TEST_PREDICTIONS_FILE)
            expected = {"Actual", "Bagging", "Boosting", "Stacking"}
            if not expected.issubset(preds.columns):
                st.warning("test_predictions.csv must contain: Actual, Bagging, Boosting, Stacking.")
            else:
                selected = st.selectbox("Choose model", ["Stacking", "Bagging", "Boosting"], key="plot_model")
                plot_data = preds[["Actual", selected]].dropna()
                scatter = go.Figure()
                scatter.add_trace(go.Scatter(
                    x=plot_data["Actual"], y=plot_data[selected], mode="markers",
                    name="Test predictions", marker=dict(color=MODEL_COLORS[selected], size=6, opacity=.4),
                    hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
                ))
                lo = float(min(plot_data.min()))
                hi = float(max(plot_data.max()))
                scatter.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(color=AMBER, dash="dash")))
                scatter.update_layout(xaxis_title="Actual price", yaxis_title="Predicted price", height=390)
                st.plotly_chart(style_plot(scatter), use_container_width=True)
                st.caption("Only held-out test observations should appear here. The diagonal line indicates perfect prediction.")
        except Exception as exc:
            st.warning(f"Could not display the test prediction plot: {exc}")
    else:
        st.caption("Optional: export `test_predictions.csv` from your notebook to enable the actual-vs-predicted chart. This section is intentionally not generated from training data.")

with insights_tab:
    st.subheader("Explore the Sri Lankan vehicle dataset")
    st.caption("Descriptive analysis of the original listings. These charts show associations, not model feature importance or causal effects.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Original listings", f"{len(raw):,}")
    m2.metric("Vehicle brands", f"{raw['Brand'].nunique():,}")
    m3.metric("Models in dataset", f"{raw['Model'].nunique():,}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Listings by brand")
        brands_count = raw["Brand"].value_counts().head(8).sort_values()
        brand_fig = go.Figure(go.Bar(
            x=brands_count.values, y=brands_count.index, orientation="h", marker_color=BLUE,
            hovertemplate="%{y}: %{x:,} listings<extra></extra>",
        ))
        brand_fig.update_layout(xaxis_title="Listings", yaxis_title="")
        st.plotly_chart(style_plot(brand_fig), use_container_width=True)
    with col2:
        st.markdown("#### Vehicle price distribution")
        hist = go.Figure(go.Histogram(x=raw["Price"].dropna(), nbinsx=45, marker_color=TEAL))
        hist.update_layout(xaxis_title="Price · dataset units", yaxis_title="Listings")
        st.plotly_chart(style_plot(hist), use_container_width=True)
    st.markdown("#### Average listed price by manufacturing year")
    annual = raw.groupby("YOM", as_index=False)["Price"].agg(["mean", "count"]).reset_index()
    annual = annual.loc[annual["count"] >= 10].sort_values("YOM")
    years_fig = go.Figure(go.Scatter(
        x=annual["YOM"], y=annual["mean"], mode="lines+markers", line=dict(color=TEAL, width=3),
        marker=dict(size=6), customdata=annual["count"],
        hovertemplate="Year: %{x}<br>Average price: %{y:.2f}<br>Listings: %{customdata}<extra></extra>",
    ))
    years_fig.update_layout(xaxis_title="Manufacturing year", yaxis_title="Average price · dataset units")
    st.plotly_chart(style_plot(years_fig), use_container_width=True)
    st.caption("Years with fewer than ten listings are hidden to reduce unstable averages.")

with about_tab:
    st.subheader("Methodology and limitations")
    st.markdown("<div class='steps'><p><b>01 · Dataset</b><br>Sri Lankan vehicle listings; original file contains 9,788 records and 17 columns.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='steps'><p><b>02 · Preprocessing</b><br>Remove saved index and duplicate rows, drop target price and listing date, split 80/20, one-hot encode categories, scale numerical fields.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='steps'><p><b>03 · Bagging</b><br>Random Forest Regressor aggregates many independently trained trees.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='steps'><p><b>04 · Boosting</b><br>Gradient Boosting Regressor adds trees sequentially to reduce residual errors.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='steps'><p><b>05 · Stacking</b><br>Random Forest, Gradient Boosting and KNN base regressors; Ridge Regression meta-model trained using five-fold out-of-fold predictions.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='steps'><p><b>06 · Evaluation</b><br>Compare MAE, RMSE and R² on an unseen held-out test set.</p></div>", unsafe_allow_html=True)
    st.warning("**Limitations:** This system estimates historical listing prices, not verified sale prices or current 2026 valuations. Expensive outliers can cause large errors, and the original price units are not independently documented.")
    st.caption("AI assistance: ChatGPT helped with code guidance and dashboard design. The data processing, model training, and recorded results were completed in Jupyter Notebook.")

st.divider()
st.caption("AUTOVALUE SL · Ensemble Learning Assignment · Educational use only")
