import streamlit as st
import joblib
import pandas as pd

# Load Model and Scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Hospital Treatment Cost Estimation",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital Treatment Cost Estimation System")
st.write("Predict the estimated medical treatment cost using Machine Learning.")

st.markdown("---")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=25)

sex = st.selectbox("Gender", ["Female", "Male"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

smoker = st.selectbox("Smoker", ["No", "Yes"])

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

# Encoding
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0

region_dict = {
    "Northeast":0,
    "Northwest":1,
    "Southeast":2,
    "Southwest":3
}

region = region_dict[region]

if st.button("Predict Treatment Cost"):

    data = pd.DataFrame(
        [[age, sex, bmi, children, smoker, region]],
        columns=[
            "age",
            "sex",
            "bmi",
            "children",
            "smoker",
            "region"
        ]
    )

    scaled = scaler.transform(data)

    prediction = model.predict(data)

    st.success(
        f"Estimated Treatment Cost : ₹ {prediction[0]:,.2f}"
    )

st.markdown("---")
st.write("Developed using Python, Scikit-learn & Streamlit")