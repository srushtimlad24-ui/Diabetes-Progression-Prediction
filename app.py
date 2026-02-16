# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Linear Regression on Diabetes Dataset")

# -----------------------------
# Load Dataset
# -----------------------------

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared Score: {r2:.2f}")

# -----------------------------
# Visualization
# -----------------------------

st.subheader("Model Visualizations")

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# True vs Predicted
axs[0].scatter(y_test, y_pred, alpha=0.5)
axs[0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2
)
axs[0].set_title("True vs Predicted Values")
axs[0].set_xlabel("True Values")
axs[0].set_ylabel("Predicted Values")
axs[0].grid(True)

# BMI vs Predicted
axs[1].scatter(X_test[:, 2], y_pred, alpha=0.7)
axs[1].set_title("BMI vs Predicted Diabetes Progression")
axs[1].set_xlabel("BMI (Feature Index 2)")
axs[1].set_ylabel("Predicted Value")
axs[1].grid(True)

plt.tight_layout()

st.pyplot(fig)
