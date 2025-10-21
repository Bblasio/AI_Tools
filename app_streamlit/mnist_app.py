# ============================================
# 🧠 Streamlit App - MNIST Digit Classifier
# ============================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------
# Load trained model (from your saved .h5 file)
# --------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

# --------------------------------------------
# App Layout
# --------------------------------------------
st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload a **28x28 grayscale image** of a handwritten digit (0–9) to predict its value.")

# Sidebar Info
st.sidebar.header("📊 Model Info")
st.sidebar.write("**Framework:** TensorFlow / Keras")
st.sidebar.write("**Dataset:** MNIST")
st.sidebar.write("**Model Type:** Convolutional Neural Network (CNN)")
st.sidebar.write("**Expected Accuracy:** ~95–99%")

# --------------------------------------------
# File Uploader
# --------------------------------------------
uploaded_file = st.file_uploader("Upload a digit image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(image)
    st.image(image, caption="Uploaded Image", width=150)

    # Normalize and reshape
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(i
