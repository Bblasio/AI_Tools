# ============================================
# 🧠 TASK 2: Deep Learning with TensorFlow
# Dataset: MNIST Handwritten Digits
# ============================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title("🧮 MNIST Digit Classifier (TensorFlow CNN)")

# Step 1: Load MNIST Dataset
st.write("### Loading Dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to add channel dimension (28x28x1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Step 2: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model Button
if st.button("🚀 Train CNN Model"):
    with st.spinner("Training model... please wait ⏳"):
        history = model.fit(x_train, y_train, epochs=3,
                            validation_data=(x_test, y_test), verbose=0)
    st.success("✅ Model training completed!")

    # Step 4: Evaluate Model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"### ✅ Test Accuracy: {test_acc*100:.2f}%")

    # Step 5: Visualize Training Performance
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Training vs Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

# Step 6: Test Prediction
st.write("### Try it out — Test the Model on a Random Image")

if st.button("🔍 Predict Random Digit"):
    idx = np.random.randint(0, len(x_test))
    sample = x_test[idx].reshape(1, 28, 28, 1)
    prediction = model.predict(sample)
    predicted_label = np.argmax(prediction)

    fig, ax = plt.subplots()
    ax.imshow(x_test[idx].reshape(28,28), cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
    st.write(f"### Predicted Label: {predicted_label}")
