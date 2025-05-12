import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)

# Display first few rows
df.head()

# Check for missing values and data types
print(df.info())
print(df.describe())

# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Convert Boolean values to integer (1/0)
df_encoded = df_encoded.astype(int)

df_encoded.head(10)

# Visualizing correlations
plt.figure(figsize=(8,6))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Define independent (X) and dependent (y) variables
X = df_encoded[['age']]
y = df_encoded['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Simple Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on test data
y_pred = lin_reg.predict(X_test)

# Compute R² and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Simple Linear Regression R²: {r2:.4f}')
print(f'Simple Linear Regression RMSE: {rmse:.2f}')

# Define independent (X) and dependent (y) variables
# Add the single variable you believe will be more predictive than age
X = df_encoded[['smoker_yes']]
y = df_encoded['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Simple Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predict on test data
y_pred = lin_reg.predict(X_test)

# Compute R² and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Simple Linear Regression R²: {r2:.4f}')
print(f'Simple Linear Regression RMSE: {rmse:.2f}')


# Define X and y for multiple regression
X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multiple Linear Regression model
multi_reg = LinearRegression()
multi_reg.fit(X_train, y_train)

# Predict on test data
y_pred_multi = multi_reg.predict(X_test)

# Compute R²
r2_multi = r2_score(y_test, y_pred_multi)

print(f'Multiple Linear Regression R²: {r2_multi:.4f}')

# Compute correlation matrix for multiple regression model
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Feature Correlations')
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

import statsmodels.api as sm

# Define X (all variables except bmi)
X_other = X.drop(columns=['bmi'])
X_other = sm.add_constant(X_other)  # Add intercept term

# Define y (bmi as dependent variable)
y_bmi = X['bmi']

# Fit regression model
model = sm.OLS(y_bmi, X_other).fit()

# Print R² value
print(f"R² for predicting bmi: {model.rsquared:.4f}")


# Original model with all predictors
multi_reg = LinearRegression()
multi_reg.fit(X_train, y_train)
y_pred_multi = multi_reg.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)

# Model without bmi
X_train_no_bmi = X_train.drop(columns=['bmi'])
X_test_no_bmi = X_test.drop(columns=['bmi'])

multi_reg_no_bmi = LinearRegression()
multi_reg_no_bmi.fit(X_train_no_bmi, y_train)
y_pred_no_bmi = multi_reg_no_bmi.predict(X_test_no_bmi)
r2_no_bmi = r2_score(y_test, y_pred_no_bmi)

# Compare results
print(f'Original Model R² (with bmi): {r2_multi:.4f}')
print(f'New Model R² (without bmi): {r2_no_bmi:.4f}')


from scipy.stats import f

# Get Sum of Squared Errors (SSE) for both models
SSE_full = np.sum((y_test - y_pred_multi) ** 2)  # Full model (with bmi)
SSE_reduced = np.sum((y_test - y_pred_no_bmi) ** 2)  # Reduced model (without bmi)

# Compute degrees of freedom
n = len(y_test)  # Number of observations
p_full = X_train.shape[1]  # Number of predictors in full model
p_reduced = X_train_no_bmi.shape[1]  # Number of predictors in reduced model

# Compute F-statistic
df_num = p_full - p_reduced  # Degrees of freedom for numerator
df_den = n - p_full  # Degrees of freedom for denominator
F_stat = ((SSE_reduced - SSE_full) / df_num) / (SSE_full / df_den)

# Compute p-value
p_value = 1 - f.cdf(F_stat, df_num, df_den)

# Print results
print(f"F-statistic: {F_stat:.4f}")
print(f"p-value: {p_value:.4f}")

