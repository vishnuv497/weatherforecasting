import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("weather_data.csv")  # Ensure this file contains historical weather data
    return df

df = load_data()

# Train ML Model
@st.cache_resource
def train_model(df):
    X = df[['temperature', 'humidity', 'pressure']]  # Features
    y = df['future_temperature']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

model = train_model(df)

# Streamlit UI
st.title("Weather Forecasting with AI 🌦️")
st.write("Predict temperature based on current weather conditions.")

# User Input
st.sidebar.header("Enter Weather Conditions")
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=20.0)
humidity = st.sidebar.slider("Humidity (%)", min_value=0, max_value=100, value=50)
pressure = st.sidebar.number_input("Pressure (hPa)", min_value=900, max_value=1100, value=1013)

# Predict Future Temperature
if st.sidebar.button("Predict"):
    input_data = np.array([[temperature, humidity, pressure]])
    prediction = model.predict(input_data)
    st.write(f"Predicted Future Temperature: {prediction[0]:.2f}°C")

# Fetch Real-Time Weather Data from API
API_KEY = "your_openweathermap_api_key"  # Replace with your API key
city = st.text_input("Enter City Name", "New York")

if st.button("Get Real-Time Weather"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        st.write(f"🌡 Temperature: {data['main']['temp']}°C")
        st.write(f"💧 Humidity: {data['main']['humidity']}%")
        st.write(f"🌪 Pressure: {data['main']['pressure']} hPa")
    else:
        st.error("City not found!")

