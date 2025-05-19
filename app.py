import streamlit as st
import joblib
import numpy as np

# Load model, scaler, and label encoder
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('country_label_encoder.pkl')  # LabelEncoder used for countries

# Title
st.title("🌍 Life Expectancy Predictor")
st.write("Enter the following details to predict life expectancy:")

# Country selection dropdown
country_list = list(label_encoder.classes_)
selected_country = st.selectbox("Select Country", country_list)
country_code = label_encoder.transform([selected_country])[0]

# Input fields
status = st.selectbox("Status", ["Developing", "Developed"])
adult_mortality = st.number_input("Adult Mortality (per 1000)", min_value=0.0)
infant_deaths = st.number_input("Infant Deaths", min_value=0)
alcohol = st.number_input("Alcohol Consumption (litres)", min_value=0.0)
expenditure = st.number_input("Percentage Expenditure", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
under_five_deaths = st.number_input("Under-Five Deaths", min_value=0)
polio = st.number_input("Polio Immunization (%)", min_value=0.0, max_value=100.0)
diphtheria = st.number_input("Diphtheria Immunization (%)", min_value=0.0, max_value=100.0)
hiv = st.number_input("HIV/AIDS Prevalence", min_value=0.0)
gdp = st.number_input("GDP per Capita", min_value=0.0)
thinness_5_9 = st.number_input("Thinness 5-9 years (%)", min_value=0.0)
income_comp = st.number_input("Income Composition of Resources", min_value=0.0, max_value=1.0)
schooling = st.number_input("Schooling (years)", min_value=0.0)

# Encode 'Status'
status_code = 0 if status == "Developing" else 1

# Prediction button
if st.button("🔮 Predict Life Expectancy"):
    input_data = np.array([[
        country_code, status_code, adult_mortality, infant_deaths, alcohol, expenditure,
        bmi, under_five_deaths, polio, diphtheria, hiv, gdp,
        thinness_5_9, income_comp, schooling
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"✅ Predicted Life Expectancy: **{prediction:.2f} years**")
