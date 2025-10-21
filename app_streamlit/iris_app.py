# ============================================
# 🌼 Streamlit App - Iris Flower Classifier
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Title
st.title("🌸 Iris Flower Classifier using Decision Tree")
st.write("Predict the species of Iris flower based on petal and sepal measurements.")

# Step 1: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Step 2: Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Step 3: Evaluate Model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')

st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**Accuracy:** {acc:.4f}")
st.sidebar.write(f"**Precision:** {prec:.4f}")
st.sidebar.write(f"**Recall:** {rec:.4f}")

# Step 4: User Input
st.header("🌿 Input Flower Measurements")

sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Step 5: Predict
prediction = clf.predict(user_data)
predicted_class = target_names[prediction[0]]

st.subheader("🌼 Predicted Species:")
st.success(predicted_class.capitalize())

# Step 6: Show Decision Tree Visualization
st.subheader("🌳 Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=target_names)
st.pyplot(fig)

# Step 7: Sample Predictions
st.subheader("🔍 Sample Predictions from Test Set")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison.replace({0:'setosa', 1:'versicolor', 2:'virginica'}, inplace=True)
st.dataframe(comparison.head(10))
