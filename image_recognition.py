import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle

def preprocess_image(image):
    resized_image = image.resize((28, 28))

    grayscale_image = resized_image.convert('L')

    flattened_image = np.array(grayscale_image).flatten()

    return flattened_image

def load_model(uploaded_model):
    try:
        with open(uploaded_model, 'rb') as f:
            model = pickle.load(f)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model: {e}")
        model = None

    return model

def predict_class(image, model):
    processed_image = preprocess_image(image)

    input_image = processed_image.reshape(1, 784)

    input_image = input_image / 255.0

    predictions = model.predict(input_image)

    predicted_class = np.argmax(predictions)

    return predicted_class

def main():
    st.title("Image Transformation and Prediction")

    uploaded_model = st.file_uploader( "Upload the joblib model", type=["pkl"])

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_model is not None and uploaded_file is not None:
        model = load_model(uploaded_model)

        if model is not None:
            pil_image = Image.open(uploaded_file)

            st.header("Original Image")
            st.image(pil_image, use_column_width=True)

            predicted_class = predict_class(pil_image, model)

            st.header("Predicted Class")
            st.write(f"The predicted class is: {predicted_class}")

if __name__ == "__main__":
    main()

