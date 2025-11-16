# EV Adoption Forecaster

This project is a machine learning–powered Streamlit dashboard that forecasts Electric Vehicle (EV) adoption for all counties in Washington State. It predicts EV adoption for the next 36 months and provides clear visual insights through a professional dark-themed dashboard.

Live Application:  
https://evforecast-yjwknxyldkpgtnqgqwekzg.streamlit.app/

Repository:  
https://github.com/Pratham31-coder/evforecast

---

## 1. Overview

The EV Adoption Forecaster provides county-level predictions of EV adoption trends using historical EV registration data. It enables users to:

- View cumulative EV forecast for any county  
- Compare trends across multiple counties  
- Analyze long-term adoption growth  
- Understand historical trends vs predicted values  

The application uses a clean, modern, dashboard-style user interface designed for clarity and ease of use.

---

## 2. Features

### 2.1 County-Level Forecasting
Users can select a county to view:
- Historical EV adoption data  
- Forecasted EV adoption for the next 36 months  
- Cumulative EV growth trends  

### 2.2 Interactive Visualization
The dashboard provides:
- Historical and forecasted cumulative EV trends  
- Matplotlib-based clean dark charts  
- Easy interpretation of long-term EV growth  

### 2.3 Multi-County Comparison
Users can compare up to three counties simultaneously.  
The comparison shows:
- Cumulative EV growth  
- Forecasted trends  
- Percentage growth over three years  

### 2.4 Professional User Interface
The application includes:
- Sidebar selection panel  
- Organized layout  
- Modern dark theme  
- Clear typography and spacing  

---

## 3. Machine Learning Approach

The forecasting model is built using a regression-based pipeline with the following engineered features:

- Lag features (1-month, 2-month, 3-month)  
- Rolling means  
- Percentage change metrics  
- Cumulative growth slope  
- Months since the start  
- Encoded county identifiers  

Output: Monthly EV adoption forecasts for the next 36 months.

Model file:  
forecasting_ev_model.pkl

---

## 4. Technology Stack

Frontend:
- Streamlit  
- Custom CSS  

Backend:
- Python  
- Pandas  
- NumPy  

Machine Learning:
- Scikit-learn  
- Joblib  
- Regression-based forecasting  

Deployment:
- Streamlit Community Cloud  
- GitHub integration  

---

## 5. Project Structure

├── app.py # Main Streamlit application
├── forecasting_ev_model.pkl # Trained ML model
├── preprocessed_ev_data.csv # Dataset with historical EV data
├── be6.png # Image used in dashboard
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 6. Running the Project Locally

Step 1: Clone the repository  
git clone https://github.com/Pratham31-coder/evforecast
cd evforecast
Step 2: Install dependencies  
Step 3: Run the application  

Local URL: http://localhost:8501/

---

## 7. Future Improvements

- Add interactive Plotly visualizations  
- Add a geographic EV distribution heatmap  
- Include confidence intervals for forecast values  
- Deploy on AWS/GCP for scalability  
- Add download/export options  

---

## 8. Author

Pratham Lahoti  
B.Tech CSE (AI and ML)  
Vardhaman College of Engineering  
Machine Learning and Data Science Enthusiast  

---



