"""California House Price Predictor — Streamlit app.

Deploy: push this folder to GitHub, then go to share.streamlit.io
-> New app -> pick the repo/branch -> Main file path: streamlit/app.py -> Deploy.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 California House Price Predictor")
st.write("Adjust the district characteristics in the sidebar and the trained **Ridge regression** model "
         "estimates the median house value. Compare all six models in the tab below.")

FEATURES = [
    ("MedInc", "Median income (x $10,000)", 0.5, 15.0, 3.9, 0.1),
    ("HouseAge", "Median house age (years)", 1.0, 52.0, 29.0, 1.0),
    ("AveRooms", "Average rooms per house", 0.8, 12.0, 5.4, 0.1),
    ("AveBedrms", "Average bedrooms per house", 0.3, 3.0, 1.1, 0.05),
    ("Population", "Area population", 300.0, 35000.0, 1400.0, 100.0),
    ("AveOccup", "Average occupancy", 0.6, 12.0, 3.0, 0.1),
    ("Latitude", "Latitude", 32.5, 42.0, 35.6, 0.1),
    ("Longitude", "Longitude", -124.5, -114.0, -119.6, 0.1),
]

@st.cache_resource
def train_all():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=1.0),
        "Lasso (L1)": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        rows.append({"Model": name,
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                     "R2": r2_score(y_test, pred)})
    return models["Ridge (L2)"], scaler, pd.DataFrame(rows).sort_values("R2", ascending=False)

with st.spinner("Training models (first run only)..."):
    ridge, scaler, results_df = train_all()

tab_predict, tab_models = st.tabs(["🔮 Predict a price", "📊 Compare models"])

with tab_predict:
    st.sidebar.header("District characteristics")
    values = [st.sidebar.slider(label, min_v, max_v, def_v, step)
              for _, label, min_v, max_v, def_v, step in FEATURES]
    x = np.array(values).reshape(1, -1)
    pred = float(ridge.predict(scaler.transform(x))[0])
    price = pred * 100_000
    st.metric("Predicted median house value", f"${price:,.0f}")
    st.caption("Ridge regression (L2), trained on 16,512 California districts. "
               "Target was in $100,000s — shown here converted to dollars.")
    st.bar_chart(pd.DataFrame({"value": values},
                              index=[f[0] for f in FEATURES])["value"])

with tab_models:
    st.subheader("Test-set scores (4,128 unseen districts)")
    st.dataframe(results_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
                 use_container_width=True)
    st.bar_chart(results_df.set_index("Model")["R2"])
    st.caption("Takeaway: ensembles (Random Forest, Gradient Boosting) clearly beat the linear "
               "baseline here — but the linear model is far easier to explain.")
