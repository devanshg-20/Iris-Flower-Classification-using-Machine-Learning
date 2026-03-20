import streamlit as st
import joblib
import numpy as np

# Load models
nb_model = joblib.load("naive_bayes_model.pkl")

st.title("🌸 Iris Flower Prediction App")


st.write("Enter flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = nb_model.predict(input_data)

    st.success(f"Predicted Class: {prediction[0]}")