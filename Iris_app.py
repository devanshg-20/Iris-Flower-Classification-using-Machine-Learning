import streamlit as st
import joblib
import numpy as np

# Load model
nb_model = joblib.load("naive_bayes_model.pkl")

st.title("🌸 Iris Flower Prediction App")

st.write("Enter flower measurements:")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediction button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = nb_model.predict(input_data)

    predicted_class = prediction[0]

    st.success(f"🌸 Predicted Class: {predicted_class}")

    # Map prediction to images
    image_map = {
        "Iris-setosa": "Iris_images/setosa.jpg",
        "Iris-versicolor": "Iris_images/versicolor.jpg",
        "Iris-virginica": "Iris_images/virginica.jpg"
    }

    # Show image
    if predicted_class in image_map:
        st.image(image_map[predicted_class], caption=predicted_class, use_column_width=True)
