import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Load model and scaler
# -------------------------------------------------
model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.pkl")

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("Tesla Stock Price Prediction 📈")

uploaded_file = st.file_uploader("Upload Tesla Stock CSV", type=["csv"])
days = st.selectbox("Predict Future Days", [1, 5, 10])

# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict_future(model, scaled_data, window_size, days):
    predictions = []
    last_window = scaled_data[-window_size:]

    for _ in range(days):
        x_input = last_window.reshape(1, window_size, 1)
        next_pred = model.predict(x_input, verbose=0)
        predictions.append(next_pred[0, 0])

        # slide window forward
        last_window = np.append(last_window[1:], next_pred, axis=0)

    return np.array(predictions)

# -------------------------------------------------
# Run prediction
# -------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Date handling
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Use closing price
    close_prices = df[['Close']]

    # Scale data
    scaled_data = scaler.transform(close_prices)

    # Predict future prices
    future_preds = predict_future(
        model=model,
        scaled_data=scaled_data,
        window_size=60,
        days=days
    )

    # Inverse scale
    future_prices = scaler.inverse_transform(
        future_preds.reshape(-1, 1)
    )

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    st.subheader(f"{days}-Day Predicted Closing Prices")
    st.write(future_prices)

    # Plot
    st.line_chart(future_prices)
