from __future__ import annotations

import joblib
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import os

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    file_path = os.path.join(os.path.dirname(__file__), "linear_regression_model.joblib")
    model = joblib.load(file_path)
    prediction = model.predict(np.array([[X]]))
    return prediction[0]


def create_streamlit_app():
    st.title("Final Project AI Fundamentals")

    input_feature = st.slider("Input Feature for Prediction", min_value=-3.0, max_value=3.0, value=1.27, step=0.01)

    if st.button("Predict value"):
        predicted_target = load_and_predict(X=input_feature, filename="linear_regression_model.joblib")
        st.write(f"Prediction: {predicted_target}")
        visualize_difference(input_feature, predicted_target)


def visualize_difference(input_feature: float, prediction: ArrayLike):
    base_dir = os.path.dirname(__file__)
    X_filename = os.path.join(base_dir, "X.joblib")
    y_filename = os.path.join(base_dir, "y.joblib")

    X = load(X_filename)
    y = load(y_filename)

    closest_idx = _index_of_closest(X, input_feature)
    actual_x = float(X[closest_idx][0])
    actual_target = float(y[closest_idx])

    difference = actual_target - float(prediction)

    fig = plt.figure(figsize=(6, 4))

    plt.scatter(X, y, color='grey', label='Dataset', alpha=0.6)
    plt.scatter(actual_x, actual_target, color='blue', label='Actual Target', zorder=5)
    plt.scatter(input_feature, prediction, color='red', label='Predicted Target', zorder=5)

    plt.plot([input_feature, actual_x], [prediction, actual_target], 'k--')

    plt.annotate(f'Difference = {difference:.2f}',
                 xy=(input_feature, prediction),
                 xytext=(10, 15),
                 textcoords='offset points',
                 color='black')

    plt.legend()
    plt.title('Prediction vs Actual Target')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.grid(True)

    st.pyplot(fig)


def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()