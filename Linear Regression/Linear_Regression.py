import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

# Set random seed for reproducibility
np.random.seed(42)

# Generate random square footage values (800 - 4000 sqft)
square_footage = np.random.randint(800, 4000, 500)

# Generate house prices with some randomness (base price + noise)
house_prices = 50000 + 150 * square_footage + np.random.normal(0, 50000, 500)

# Create a DataFrame
df = pd.DataFrame({
    'SquareFootage': square_footage,
    'MedianHousePrice': house_prices
})

# Display dataset preview
df.head()

plt.figure(figsize=(6,4))
sns.scatterplot(x=df['SquareFootage'], y=df['MedianHousePrice'], alpha=0.5)
plt.xlabel('Square Footage')
plt.ylabel('Median House Price')
plt.title('Linearity Check: Price vs. Square Footage')
plt.show()

# Split data
X = df[['SquareFootage']]
y = df['MedianHousePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# Generate residuals
residuals = y_test - y_pred

# Histogram and Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, bins=30, kde=True, ax=axes[0])
axes[0].set_title('Histogram of Residuals')
sm.qqplot(residuals, line='s', ax=axes[1])
axes[1].set_title('Q-Q Plot of Residuals')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='red', linestyle='dashed')
plt.xlabel('Predicted House Prices')
plt.ylabel('Residuals')
plt.title('Homoskedasticity Check')
plt.show()

# Independence of Residuals Check using Durbin-Watson Test
dw_stat = durbin_watson(residuals)
print(f'Durbin-Watson Statistic: {dw_stat:.4f}')

# Compute R^2
r2 = r2_score(y_test, y_pred)
print(f'R-Squared: {r2:.4f}')

# Visualizing Actual vs. Predicted Prices
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()


