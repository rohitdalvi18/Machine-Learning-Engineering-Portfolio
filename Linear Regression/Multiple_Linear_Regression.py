import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Small dataset (3 points) for manual computation
# In X, we include a row of 1s for the beta_0 intercept
# In X, we also have the values of X
# Here, we can imagine X values to be square footage of a house
X = np.array([[1, 0.5], [1, 1.0], [1, 1.5], [1, 2.0]])
# Here we can imagine Y values to be home prices
Y = np.array([[300], [350], [400], [450]])

X_transpose = X.T
print(X_transpose)

# Compute X^T X
# In numpy the @ is shorthand for matrix multiplication
# In numpy .T computes the transpose of a given matrix
XTX = X_transpose @ X
print("X^T X:")
print(XTX)

# Compute X^T Y
XTY = X.T @ Y
print("\nX^T Y:")
print(XTY)

# Compute (X^T X)^-1
XTX_inv = inv(XTX)
print("\n(X^T X)^-1:")
print(XTX_inv)

# Compute beta coefficients
beta_hat = XTX_inv @ XTY
print("\nComputed Beta Coefficients:")
print(beta_hat)

# Verify with sklearn
model = LinearRegression(fit_intercept=False)
model.fit(X, Y)
print("\nSklearn Beta Coefficients:")
print(model.coef_.T)

# Generate predictions using both models
X_plot = np.linspace(0, 2.2, 100).reshape(-1, 1)  # Extend X-axis to 0
X_plot_with_intercept = np.hstack((np.ones_like(X_plot), X_plot))  # Add intercept term

# Predictions
y_manual = X_plot_with_intercept @ beta_hat  # Manually computed model
y_sklearn = model.predict(X_plot_with_intercept)  # Sklearn model

# Scatter plot of original data
plt.scatter(X[:, 1], Y, color='red', label='Original Data')

# Plot regression lines
plt.plot(X_plot, y_manual, color='blue', linestyle='dashed', label='Manual Regression Line')
plt.plot(X_plot, y_sklearn, color='green', linestyle='solid', label='Sklearn Regression Line')

# Labels and legend
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.title('Comparison of Manual vs. Sklearn Regression')
plt.legend()
plt.show()

# Load Ames Housing Dataset from OpenML
df = pd.read_csv("https://raw.githubusercontent.com/wblakecannon/ames/refs/heads/master/data/housing.csv")

# Select relevant features (Square Footage, Bedrooms, and SalePrice)
df = df[['Gr Liv Area', 'Bedroom AbvGr', 'SalePrice']].dropna()

# Rename columns for clarity
df.columns = ['SquareFootage', 'Bedrooms', 'MedianHousePrice']

# Introduce Heteroskedasticity: Noise scales with house price
np.random.seed(42)
heteroskedastic_noise = np.random.normal(0, 0.40 * df['MedianHousePrice'])

# Ensure house prices stay positive by setting a minimum price floor
df['MedianHousePrice_H'] = np.maximum(df['MedianHousePrice'] + heteroskedastic_noise, 50000)



# Split data
X = df[['SquareFootage', 'Bedrooms']]
y = df['MedianHousePrice_H']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

# Calculate R² for the log-transformed model
r2_log = r2_score(y_test, y_pred)

print(f"R²: {r2_log:.4f}")

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=y_test, alpha=0.5, label="Actual vs. Predicted")
sns.lineplot(x=y_pred, y=y_pred, color='red', label="Regression Line")
plt.xlabel("Predicted House Prices")
plt.ylabel("Actual House Prices")
plt.title("Linear Model: Original Data")
plt.legend()
plt.show()

# Create a grid for visualization
sqft_range = np.linspace(df['SquareFootage'].min(), df['SquareFootage'].max(), 30)
bedroom_range = np.linspace(df['Bedrooms'].min(), df['Bedrooms'].max(), 30)
sqft_grid, bedroom_grid = np.meshgrid(sqft_range, bedroom_range)
X_grid = np.column_stack([sqft_grid.ravel(), bedroom_grid.ravel()])

# Predict prices over the grid
y_grid_pred = model.predict(X_grid)
y_grid_pred = y_grid_pred.reshape(sqft_grid.shape)

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data points
ax.scatter(df['SquareFootage'], df['Bedrooms'], df['MedianHousePrice'], color='red', label="Actual Data", alpha=0.6)

# Plot regression plane
ax.plot_surface(sqft_grid, bedroom_grid, y_grid_pred, cmap='viridis', alpha=0.5)

# Labels
ax.set_xlabel("Square Footage")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Median House Price")
ax.set_title("3D Regression Model: House Price vs. Square Footage & Bedrooms")

plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=df['SquareFootage'], y=df['MedianHousePrice_H'], ax=axes[0])
axes[0].set_title("Square Footage vs. House Price")

sns.scatterplot(x=df['Bedrooms'], y=df['MedianHousePrice_H'], ax=axes[1])
axes[1].set_title("Bedrooms vs. House Price")
plt.show()

X_with_const = sm.add_constant(X_train)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("\nVariance Inflation Factor (VIF) BEFORE transformation:\n", vif_data)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, bins=30, ax=axes[0])
axes[0].set_title("Histogram of Residuals (Before Translation)")

sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].set_title("Q-Q Plot of Residuals (Before Translation)")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel("Predicted House Prices")
plt.ylabel("Residuals")
plt.title("Homoskedasticity Check (Before Transformation)")
plt.show()

df['LogPrice'] = np.log(df['MedianHousePrice_H'])

# Split the log-transformed data
y_log = df['LogPrice']
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Fit a linear regression model on the log-transformed data
model_log = LinearRegression()
model_log.fit(X_train_log, y_train_log)
y_pred_log = model_log.predict(X_test_log)

# Calculate residuals for the log-transformed model
residuals_log = y_test_log - y_pred_log

# Calculate R² for the log-transformed model
r2_log = r2_score(y_test_log, y_pred_log)

print(f"R² for Log-Transformed Model: {r2_log:.4f}")

# Calculate RMSE for the original model
rmse_original = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE (Original Model): {rmse_original:.4f}")

# Convert log predictions back to original scale
y_pred_log_original_scale = np.exp(y_pred_log)
y_test_log_original_scale = np.exp(y_test_log)

# Calculate RMSE in original scale
rmse_log_original = np.sqrt(mean_squared_error(y_test_log_original_scale, y_pred_log_original_scale))
print(f"RMSE (Log Model in Original Scale): {rmse_log_original:.4f}")

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred_log, y=y_test_log, alpha=0.5, label="Actual vs. Predicted")
sns.lineplot(x=y_pred_log, y=y_pred_log, color='red', label="Regression Line")
plt.xlabel("Predicted Log House Prices")
plt.ylabel("Actual Log House Prices")
plt.title("Linear Model: Log-Transformed Data")
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=df['SquareFootage'], y=df['LogPrice'], ax=axes[0])
axes[0].set_title("Square Footage vs. Log(House Price)")

sns.scatterplot(x=df['Bedrooms'], y=df['LogPrice'], ax=axes[1])
axes[1].set_title("Bedrooms vs. Log(House Price)")
plt.show()

X_with_const_log = sm.add_constant(X_train_log)
vif_data_log = pd.DataFrame()
vif_data_log["Feature"] = X_with_const_log.columns
vif_data_log["VIF"] = [variance_inflation_factor(X_with_const_log.values, i) for i in range(X_with_const_log.shape[1])]
print("\nVariance Inflation Factor (VIF) AFTER transformation:\n", vif_data_log)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals_log, kde=True, bins=30, ax=axes[0])
axes[0].set_title("Histogram of Residuals (Log)")

sm.qqplot(residuals_log, line='s', ax=axes[1])
axes[1].set_title("Q-Q Plot of Residuals (Log)")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred_log, y=residuals_log)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel("Predicted Log House Prices")
plt.ylabel("Residuals")
plt.title("Homoskedasticity Check (After Transformation)")
plt.show()

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Ames Housing Dataset from OpenML
df = pd.read_csv("https://raw.githubusercontent.com/wblakecannon/ames/refs/heads/master/data/housing.csv")

# Select relevant features (Square Footage, Bedrooms, and SalePrice)
df = df[['Gr Liv Area', 'Bedroom AbvGr', 'SalePrice']].dropna()

# Rename columns for clarity
df.columns = ['SquareFootage', 'Bedrooms', 'MedianHousePrice']

# Display first few rows
df.head()

print(df.describe())
sns.pairplot(df)
plt.show()

X = df[['SquareFootage', 'Bedrooms']]
y = df['MedianHousePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

simple_model = LinearRegression()
simple_model.fit(X_train[['SquareFootage']], y_train)

y_pred_simple = simple_model.predict(X_test[['SquareFootage']])
r2_simple = r2_score(y_test, y_pred_simple)

print(f"Simple Linear Regression R²: {r2_simple:.4f}")

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

y_pred_multi = multi_model.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)

print(f"Multiple Linear Regression R²: {r2_multi:.4f}")

print(f"R² Score for Multiple Linear Regression: {r2_multi:.4f}")

rmse_multi = mean_squared_error(y_test, y_pred_multi)
print(f"RMSE for Multiple Linear Regression: ${np.sqrt(rmse_multi):,.2f}")

print(f"Simple Linear Regression R²: {r2_simple:.4f}")
print(f"Multiple Linear Regression R²: {r2_multi:.4f}")

plt.figure(figsize=(10, 5))

# Simple Regression Plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_simple, alpha=0.5, label="Predicted vs. Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Simple Linear Regression")
plt.legend()

# Multiple Regression Plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_multi, alpha=0.5, label="Predicted vs. Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Multiple Linear Regression")
plt.legend()

plt.tight_layout()
plt.show()

improvement = r2_multi - r2_simple
print(f"R² Improvement: {improvement:.4f}")
