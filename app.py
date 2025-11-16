import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="EV Adoption Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- GLOBAL CSS STYLE ----------------------
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #0e1117 !important;
    color: white !important;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    margin-top: 10px;
    color: white;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    font-weight: 400;
    text-align: center;
    color: #c7c7c7;
    margin-bottom: 30px;
}

/* Card styling */
.card {
    background-color: #161a23;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 0px 10px rgba(255,255,255,0.05);
    margin-bottom: 20px;
}

/* Text inside dropdowns */
.css-16idsys p {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE SECTION ----------------------
st.markdown("<div class='title'>🔮 EV Adoption Forecaster</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Forecast 3-year EV growth for any county in Washington</div>", unsafe_allow_html=True)

# -------------------- LOAD MODEL -------------------------
model = joblib.load('forecasting_ev_model.pkl')

# -------------------- LOAD DATA --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

county_list = sorted(df['County'].unique())

# -------------------- LAYOUT: SIDEBAR ---------------------
with st.sidebar:
    st.header("⚙️ Select County")
    county = st.selectbox("Choose a County", county_list)
    st.image("be6.png")
    st.markdown("---")
    st.caption("Designed for the AICTE Internship Cycle 2")

# -------------------- ORIGINAL FORECAST LOGIC (UNCHANGED) ---------------------
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

forecast_df = pd.DataFrame(future_rows)
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + cumulative_ev[-1]

# -------------------- METRICS ---------------------
latest_historical_ev = cumulative_ev[-1]
latest_forecast_ev = forecast_df['Cumulative EV'].iloc[-1]

growth_pct = ((latest_forecast_ev - latest_historical_ev) / latest_historical_ev) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Current EV Count", f"{latest_historical_ev:,}")
col2.metric("Forecasted EV (3 yrs)", f"{latest_forecast_ev:,}")
col3.metric("Growth %", f"{growth_pct:.2f}%")

# -------------------- GRAPH SECTION ---------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader(f"📈 Cumulative EV Forecast for {county}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(pd.concat([county_df['Date'], forecast_df['Date']]),
        pd.concat([pd.Series(cumulative_ev), forecast_df['Cumulative EV']]),
        color="#4DB6AC",
        marker='o')

ax.set_facecolor("#0e1117")
fig.patch.set_facecolor("#0e1117")
ax.grid(True, alpha=0.2)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

st.success("Forecast complete ✔")
