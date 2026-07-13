from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """
    model = load(filename)
    y = model.predict(X)
    return y

def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.

    This function sets up a Streamlit app with a user interface for inputting a single feature
    value and making predictions using a pre-trained regression model. The app includes:

    - A title displayed at the top of the app.
    - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    - A "Predict value" button that, when clicked, triggers the prediction process.
    - Upon clicking the "Predict value" button, the function:
        - Calls `load_and_predict`, passing the selected feature as input, to load the regression model
          and make a prediction.
        - Displays the prediction result on the app.
        - Calls `visualize_difference`, passing the input feature and the prediction result,
          to visualize the difference between the predicted value and the actual value in the original dataset.

    Note: This function does not return any value. It directly manipulates the Streamlit app's UI by
    writing content and rendering UI elements.
    """
    # Streamlit app title
    st.title("Simple Regression model prediction")

    # Бонус: сайдбар з інформацією про модель, її метриками та опціями графіка
    _render_sidebar()

    # User input for new prediction using a slider
    input_feature = st.slider("Input Feature for Prediction", min_value=-3.0, max_value=3.0, value=0.0, step=0.01)

    # Button to make a prediction
    if st.button("Predict value"):
        # 1. Call load_and_predict functions.
        prediction = load_and_predict([[input_feature]])

        # 2. Display the prediction.
        st.write(f"Prediction: {prediction[0]}")

        # 3. Call visualize_difference to display a plot visualizing the difference between actual and predicted value.
        visualize_difference(input_feature, prediction[0])

def visualize_difference(input_feature: float, prediction: ArrayLike):
    """
    Deserialize and load the initial datasets. Calculate the difference between actual data
    in the 'y' dataset and the predicted value for a given 'input_feature'.

    Visualize the difference by plotting the entire 'X' & 'y' as a Scatter plot. Then add
    a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given 'input_feature'.
    Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    Args:
        input_feature (float): User provided data used for prediction.
        prediction (array-like): Predicted value.

    """
    # Load the X and y datasets
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    X = load(X_filename)

    y = load(y_filename)

    actual_target = y[_index_of_closest(X, input_feature)]

    # Calculate difference
    difference = actual_target - prediction

    # Бонус: числове порівняння "прогноз vs факт" у вигляді метрик
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual target", f"{actual_target:.2f}")
    col2.metric("Predicted target", f"{float(prediction):.2f}")
    col3.metric("Difference", f"{difference:.2f}")

    # Visualization
    fig = plt.figure(figsize=(6,4))

    # Plot the entire dataset (X, y) as grey dots to visualize the data distribution.
    plt.scatter(X, y, color="grey", label="Dataset")

    # Plot the actual target value for a specific input feature as a blue dot.
    plt.scatter(input_feature, actual_target, color="blue", label="Actual Target", zorder=3)

    # Plot the predicted target value for the same input feature as a red dot.
    plt.scatter(input_feature, prediction, color="red", label="Predicted Target", zorder=3)

    # Бонус: лінія регресії показує, як модель поводиться на всьому діапазоні ознаки
    if st.session_state.get("show_regression_line", False):
        try:
            model = load("linear_regression_model.joblib")
            x_line = np.linspace(float(np.min(X)), float(np.max(X)), 100).reshape(-1, 1)
            plt.plot(x_line, model.predict(x_line), color="tomato", linewidth=1.5, alpha=0.7, label="Regression line")
        except FileNotFoundError:
            pass

    # Display a legend on the plot to label the different scatter points (dataset, actual target, predicted target).
    plt.legend()

    # Set the title of the plot, describing what is being visualized.
    plt.title("Prediction vs Actual Target")

    # Set the label for the x-axis to 'Feature', indicating that the x-axis represents the input features.
    plt.xlabel("Feature")

    # Set the label for the y-axis to 'Target', indicating that the y-axis represents the target values (actual or predicted).
    plt.ylabel("Target")

    # Enable a grid on the plot to improve readability.
    plt.grid(True)

    # Draw a dashed line ('k--' for black dashed line) between the actual and predicted target values to visually represent the difference.
    plt.plot([input_feature, input_feature], [actual_target, prediction], "k--")

    # Annotate the plot with the difference between the actual and predicted target values, positioned halfway between them and offset slightly for visibility.
    mid_y = (actual_target + float(prediction)) / 2
    plt.annotate(f"Difference = {difference:.2f}", xy=(input_feature, mid_y), xytext=(input_feature + 0.15, mid_y))

    st.pyplot(fig)

# Бонус: сайдбар з рівнянням моделі, метриками з тестової вибірки та опціями графіка.
# Метрики зберігає model.py (metrics.joblib); якщо файлів ще немає — підказуємо, що запустити.
def _render_sidebar():
    st.sidebar.header("About the model")
    try:
        model = load("linear_regression_model.joblib")
    except FileNotFoundError:
        st.sidebar.warning("Model file not found. Run `python model.py` first.")
        return

    coef = float(np.ravel(model.coef_)[0])
    intercept = float(model.intercept_)
    sign = "+" if intercept >= 0 else "-"
    st.sidebar.latex(rf"\hat{{y}} = {coef:.2f} \cdot x {sign} {abs(intercept):.2f}")
    st.sidebar.caption("Synthetic dataset: 100 samples, 1 feature, noise=20")

    try:
        metrics = load("metrics.joblib")
        st.sidebar.subheader("Test-set metrics")
        left, right = st.sidebar.columns(2)
        for (name, value), column in zip(metrics.items(), [left, right, left, right]):
            column.metric(name, f"{value:.2f}")
    except FileNotFoundError:
        pass

    st.sidebar.subheader("Chart options")
    st.sidebar.checkbox("Show regression line", value=True, key="show_regression_line")

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates
    the absolute difference between each element in `X` and `k`, identifies the minimum value
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple,
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()
