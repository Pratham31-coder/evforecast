import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="EV Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- LOAD MODEL -----------------------
model = joblib.load('forecasting_ev_model.pkl')

# -------------------- GLOBAL STYLE (DARK DASHBOARD) ----------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #FFFFFF;
        }
        .main-title {
            font-size: 38px;
            font-weight: 800;
            text-align: center;
            margin-top: 10px;
            color: #FFFFFF;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #c7c7c7;
            margin-bottom: 25px;
        }
        .card {
            background-color: #161a23;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
            margin-bottom: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------------
st.markdown(
    "<div class='main-title'>🔮 EV Adoption Forecaster for a County in Washington State</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='sub-title'>Forecast EV adoption trends for the next 3 years at the county level.</div>",
    unsafe_allow_html=True
)

# -------------------- LOAD DATA ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

county_list = sorted(df['County'].dropna().unique().tolist())

# -------------------- SIDEBAR --------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    county = st.selectbox("Select a County", county_list)
    st.markdown("---")
    st.image("be6.png", use_container_width=True)
    st.markdown("---")
    st.caption("Prepared for the AICTE Internship Cycle 2 by S4F")

# -------------------- VALIDATION -----------------------
if county not in df['County'].unique():
    st.warning(f"County '{county}' not found in dataset.")
    st.stop()

# -------------------- FORECASTING LOGIC (UNCHANGED) ---------------------
county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = 36

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# -------------------- COMBINE HISTORICAL + FORECAST ---------------------
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# -------------------- METRICS (USES EXISTING VALUES) --------------------
historical_total = historical_cum['Cumulative EV'].iloc[-1]
forecasted_total = forecast_df['Cumulative EV'].iloc[-1]

if historical_total > 0:
    forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
else:
    forecast_growth_pct = 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Current Cumulative EVs", f"{historical_total:,}")
m2.metric("Forecasted Cumulative EVs", f"{forecasted_total:,}")
m3.metric("3-Year Growth (%)", f"{forecast_growth_pct:.2f}%")

# -------------------- MAIN FORECAST PLOT ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(f"📊 Cumulative EV Forecast for {county} County")

fig, ax = plt.subplots(figsize=(12, 6))
for label, data in combined.groupby('Source'):
    ax.plot(data['Date'], data['Cumulative EV'], label=label, marker='o')
ax.set_title(f"Cumulative EV Trend - {county} (3 Years Forecast)", fontsize=14, color='white')
ax.set_xlabel("Date", color='white')
ax.set_ylabel("Cumulative EV Count", color='white')
ax.grid(True, alpha=0.3)
ax.set_facecolor("#1c1c1c")
fig.patch.set_facecolor('#1c1c1c')
ax.tick_params(colors='white')
ax.legend()
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------- TEXT SUMMARY (UNCHANGED LOGIC) --------------------
if historical_total > 0:
    trend = "increase 📈" if forecast_growth_pct > 0 else "decrease 📉"
    st.success(
        f"Based on the graph, EV adoption in **{county}** is expected to show a **{trend} of {forecast_growth_pct:.2f}%** over the next 3 years."
    )
else:
    st.warning("Historical EV total is zero, so percentage forecast change can't be computed.")

# -------------------- MULTI-COUNTY COMPARISON (SAME LOGIC, NICER WRAP) ----------------
st.markdown("---")
st.header("Compare EV Adoption Trends for up to 3 Counties")

multi_counties = st.multiselect("Select up to 3 counties to compare", county_list, max_selections=3)

if multi_counties:
    comparison_data = []

    for cty in multi_counties:
        cty_df = df[df['County'] == cty].sort_values("Date")
        cty_code = cty_df['county_encoded'].iloc[0]

        hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
        cum_ev = list(np.cumsum(hist_ev))
        months_since = cty_df['months_since_start'].max()
        last_date = cty_df['Date'].max()

        future_rows_cty = []
        for i in range(1, forecast_horizon + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            months_since += 1
            lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            recent_cum = cum_ev[-6:]
            ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0

            new_row = {
                'months_since_start': months_since,
                'county_encoded': cty_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_slope
            }
            pred = model.predict(pd.DataFrame([new_row]))[0]
            future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

            hist_ev.append(pred)
            if len(hist_ev) > 6:
                hist_ev.pop(0)

            cum_ev.append(cum_ev[-1] + pred)
            if len(cum_ev) > 6:
                cum_ev.pop(0)

        hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()

        fc_df = pd.DataFrame(future_rows_cty)
        fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]

        combined_cty = pd.concat([
            hist_cum[['Date', 'Cumulative EV']],
            fc_df[['Date', 'Cumulative EV']]
        ], ignore_index=True)

        combined_cty['County'] = cty
        comparison_data.append(combined_cty)

    comp_df = pd.concat(comparison_data, ignore_index=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Comparison of Cumulative EV Adoption Trends")
    fig, ax = plt.subplots(figsize=(14, 7))
    for cty, group in comp_df.groupby('County'):
        ax.plot(group['Date'], group['Cumulative EV'], marker='o', label=cty)
    ax.set_title("EV Adoption Trends: Historical + 3-Year Forecast", fontsize=16, color='white')
    ax.set_xlabel("Date", color='white')
    ax.set_ylabel("Cumulative EV Count", color='white')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("#1c1c1c")
    fig.patch.set_facecolor('#1c1c1c')
    ax.tick_params(colors='white')
    ax.legend(title="County")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    
    growth_summaries = []
    for cty in multi_counties:
        cty_df = comp_df[comp_df['County'] == cty].reset_index(drop=True)
        historical_total_cty = cty_df['Cumulative EV'].iloc[len(cty_df) - forecast_horizon - 1]
        forecasted_total_cty = cty_df['Cumulative EV'].iloc[-1]

        if historical_total_cty > 0:
            growth_pct = ((forecasted_total_cty - historical_total_cty) / historical_total_cty) * 100
            growth_summaries.append(f"{cty}: {growth_pct:.2f}%")
        else:
            growth_summaries.append(f"{cty}: N/A (no historical data)")

    growth_sentence = " | ".join(growth_summaries)
    st.success(f"Forecasted EV adoption growth over next 3 years — {growth_sentence}")

st.success("Forecast complete ✅")
