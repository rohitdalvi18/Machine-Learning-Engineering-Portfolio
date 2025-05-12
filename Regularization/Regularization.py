# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Set seed for reproducibility
np.random.seed(42)

# Generate full dataset (ground truth relationship)
X_full = np.linspace(0, 2, 100).reshape(-1, 1)  # Feature
y_full = 4 + 3 * X_full + np.random.randn(100, 1) * 1.2  # Smaller noise for realism

# Sample only a few points (to force overfitting)
X_train_idx = np.random.choice(len(X_full), 5, replace=False)  # Pick 5 random training points
X_train = X_full[X_train_idx]
y_train = y_full[X_train_idx]

# Remaining points are test data
X_test = np.delete(X_full, X_train_idx).reshape(-1, 1)
y_test = np.delete(y_full, X_train_idx).reshape(-1, 1)

# Lower-degree polynomial regression to fix overfitting
# ---------------------------------------------------------
# Instead of using a very high-degree polynomial, we use degree=3 to balance flexibility and avoid extreme overfitting.
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(X_train, y_train)  # Train the polynomial regression model using the small training dataset

# Simple mean model to demonstrate underfitting
# ---------------------------------------------------------
# The model simply predicts the average target value for all inputs.
# This represents an extreme case of underfitting, where the model is too simple to capture any trend.
y_mean = np.full_like(y_full, np.mean(y_train))  # Predicts the same mean value for all X values


# Plot Overfitting vs. Underfitting
plt.figure(figsize=(12, 5))

# Overfitting: Lower-degree polynomial to fit training data
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color="blue", label="Training Data (5 Points)", edgecolor="black")
plt.scatter(X_test, y_test, color="gray", alpha=0.5, label="Test Data")
plt.plot(X_full, poly_model.predict(X_full), color="red", linestyle="dashed", label="Polynomial Fit (Overfit)")
plt.title("Overfitting: Model Captures Noise but Doesn't Generalize")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()

# Underfitting: Predicting a single mean value
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, color="blue", label="Training Data (5 Points)", edgecolor="black")
plt.scatter(X_test, y_test, color="gray", alpha=0.5, label="Test Data")
plt.plot(X_full, y_mean, color="red", linestyle="dashed", label="Mean Prediction (Underfit)")
plt.title("Underfitting: Model Ignores Pattern")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()

plt.show()

# Effect of Lambda in Ridge Regression (Coefficient Shrinkage)
# ---------------------------------------------------------
# We train multiple Ridge regression models with different levels of regularization (lambda values)
lambdas = [0.01, 1, 10, 100]  # Different lambda values to test regularization impact

plt.figure(figsize=(8, 5))  # Create a new figure with appropriate size
plt.scatter(X_full, y_full, color="blue", alpha=0.5, label="Data")  # Plot the original data points

# Train Ridge models with increasing lambda values
colors = ["green", "orange", "purple", "black"]  # Different colors for different lambda values
for i, l in enumerate(lambdas):
    ridge = Ridge(alpha=l)  # Initialize Ridge Regression with specific lambda
    ridge.fit(X_train, y_train)  # Train Ridge on the training data
    plt.plot(X_full, ridge.predict(X_full), color=colors[i], linestyle="dashed", label=f"Ridge λ={l}")
    # Plot the Ridge regression predictions across the full dataset

plt.title("Effect of Increasing Lambda in Ridge Regression")  # Add title
plt.xlabel("Feature")  # X-axis label
plt.ylabel("Target")  # Y-axis label
plt.legend()  # Show legend to distinguish different lambda values
plt.show()  # Display the plot

# Use Cross-Validation to Find the Optimal Lambda (Hyperparameter Tuning)
# ---------------------------------------------------------
ridge_cv_scores = []  # List to store cross-validation MSE scores
lambda_values = np.logspace(-3, 3, 50)  # Generate 50 lambda values ranging from 10^-3 to 10^3 (log scale)

# Loop through different lambda values and perform cross-validation
for l in lambda_values:
    ridge = Ridge(alpha=l)  # Initialize Ridge model with current lambda value
    scores = cross_val_score(ridge, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
    # Perform 5-fold cross-validation to compute MSE
    # Since MSE is a “lower is better” metric, sklearn makes it negative so that
    # higher (less negative) values indicate better performance
    ridge_cv_scores.append(np.mean(scores))  # Store the mean cross-validation score for this lambda

# Convert negative MSE scores to positive values for easier interpretation
ridge_cv_scores = np.abs(ridge_cv_scores)

# Plot the cross-validation error curve
plt.figure(figsize=(8, 5))
plt.plot(lambda_values, ridge_cv_scores, marker="o", linestyle="dashed", color="blue", label="Cross-Validation Error")
plt.xscale("log")  # Log-scale for better visualization
plt.xlabel("Lambda (Regularization Strength)")
plt.ylabel("Mean Squared Error")
plt.title("Cross-Validation to Select Optimal Lambda")
plt.legend()
plt.show()


