# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

# Scikit-learn for modeling and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Set global plotting style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

from google.colab import drive
import os
drive.mount('/content/drive', force_remount=True)

# Load and Explore the Dataset

project_folder = '/content/drive/My Drive/Colab Notebooks/'

file_path = os.path.join(project_folder, "processed_data.csv")

# Read the CSV
df = pd.read_csv(file_path)


# Define the number of bins (for quantiles) and the maximum samples per bin
num_bins = 5
n_samples_per_bin = 200

# Create a temporary binned column using pd.qcut.
# This column is only used for sampling and will be dropped later.
df['temp_fare_bin'] = pd.qcut(df['totalFare'], q=num_bins, duplicates='drop')

# For each bin, sample up to n_samples_per_bin rows (or all rows if fewer exist)
balanced_subset = (
    df.groupby('temp_fare_bin', group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), n_samples_per_bin), random_state=42))
      .reset_index(drop=True)
)

print("Balanced subset shape:", balanced_subset.shape)

# Drop the temporary binned column so that the column structure remains unchanged.
balanced_subset = balanced_subset.drop(columns=['temp_fare_bin'])

# Overwrite the original df with this balanced subset for the remainder of the analysis.
df = balanced_subset.copy()

# Display column names to verify structure
print("Columns in dataset:")
print(df.columns)

# Preview the first few rows and columns
df.head(10)

# -- Drop unnecessary columns --
# We drop identifiers, date strings, and other columns that are not used for prediction.
cols_to_drop = [
    'legId', 'searchDate', 'flightDate', 'fareBasisCode',
    'segmentsDepartureTimeEpochSeconds', 'segmentsDepartureTimeRaw',
    'segmentsArrivalTimeEpochSeconds', 'segmentsArrivalTimeRaw',
    'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode',
    'segmentsAirlineName', 'segmentsAirlineCode',
    'segmentsEquipmentDescription', 'segmentsDurationInSeconds',
    'segmentsDistance', 'segmentsCabinCode',
    'baseFare'  # Dropped to avoid collinearity with totalFare
]
df = df.drop(columns=cols_to_drop, errors='ignore')

# -- Convert boolean columns to integers --
bool_cols = ['isBasicEconomy', 'isRefundable', 'isNonStop']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# -- Convert travelDuration (ISO 8601 format) to total minutes --
# Example format: "PT5H45M"
def convert_duration_iso(duration_str):
    """
    Convert a duration string in ISO 8601 format (e.g., "PT5H45M") to total minutes.
    If the format is not recognized or the value is missing, returns np.nan.
    """
    if pd.isna(duration_str):
        return np.nan
    try:
        # Remove the "PT" prefix if present
        if duration_str.startswith("PT"):
            duration_str = duration_str[2:]
        hours = 0
        minutes = 0
        if "H" in duration_str:
            parts = duration_str.split("H")
            hours = int(parts[0])
            # The remainder should contain minutes ending with "M"
            if len(parts) > 1 and "M" in parts[1]:
                minutes = int(parts[1].split("M")[0])
        elif "M" in duration_str:
            minutes = int(duration_str.split("M")[0])
        return hours * 60 + minutes
    except Exception as e:
        return np.nan

# Create a new column 'travelDurationMinutes'
if 'travelDuration' in df.columns:
    df['travelDurationMinutes'] = df['travelDuration'].apply(convert_duration_iso)
    print("\nTravel Duration conversion preview:")
    print(df[['travelDuration', 'travelDurationMinutes']].head())
else:
    print("Column 'travelDuration' not found.")

# -- For simplicity, we drop rows with missing values in our key features --
# We'll use the following features (interpretable predictors):
selected_features = ['travelDurationMinutes', 'totalTravelDistance', 'isBasicEconomy', 'isRefundable', 'isNonStop']

# Ensure numeric columns are converted properly (e.g., totalTravelDistance)
for col in ['totalTravelDistance']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Set target variable (using totalFare)
if 'totalFare' not in df.columns:
    raise ValueError("Target column 'totalFare' not found in the dataset.")

# Combine features and target; then drop rows with NaNs in these columns
df_model = df[selected_features + ['totalFare']].replace([np.inf, -np.inf], np.nan).dropna()

# Reassign X and y
X = df_model[selected_features]
y = df_model['totalFare']

print("\nCleaned Features Sample:")
print(X.head())
print("\nCleaned Target Sample:")
print(y.head())

# EDA

print("\nSummary statistics of features:")
print(X.describe())

# Histograms for each feature
X.hist(bins=20, figsize=(12,8))
plt.suptitle("Histograms of Selected Features")
plt.show()

# Correlation heatmap among features and target
corr_data = X.copy()
corr_data['totalFare'] = y
plt.figure()
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot: travelDurationMinutes vs. totalFare
df_plot = pd.DataFrame({
    'travelDurationMinutes': X['travelDurationMinutes'],
    'totalFare': y
})
plt.figure()
sns.scatterplot(data=df_plot, x='travelDurationMinutes', y='totalFare')
plt.xlabel("Travel Duration (minutes)")
plt.ylabel("Total Fare (USD)")
plt.title("Travel Duration vs. Total Fare")
plt.show()

# Scatter plot: totalTravelDistance vs. totalFare
df_plot = pd.DataFrame({
    'totalTravelDistance': X['totalTravelDistance'],
    'totalFare': y
})
plt.figure()
sns.scatterplot(data=df_plot, x='totalTravelDistance', y='totalFare')
plt.xlabel("Travel Distance")
plt.ylabel("Total Fare (USD)")
plt.title("Travel Duration vs. Total Fare")
plt.show()

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test set
y_pred_lr = lr.predict(X_test)

# Calculate evaluation metrics: RMSE and R-squared
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Performance:")
print("RMSE:", rmse_lr)
print("R-squared:", r2_lr)

# Plot Actual vs. Predicted for Linear Regression
plt.figure()
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.7, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel("Actual Total Fare (USD)")
plt.ylabel("Predicted Total Fare (USD)")
plt.title("Linear Regression: Actual vs. Predicted")
plt.legend()
plt.show()

# Initialize and train a Decision Tree Regressor
tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Predict on test data using the regression tree
y_pred_tree = tree.predict(X_test)

# Calculate evaluation metrics for the tree model: RMSE and R-squared
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

print("\nRegression Tree Performance:")
print("RMSE:", rmse_tree)
print("R-squared:", r2_tree)

# Plot Actual vs. Predicted for the Regression Tree
plt.figure()
plt.scatter(y_test, y_pred_tree, color='green', alpha=0.7, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
plt.xlabel("Actual Total Fare (USD)")
plt.ylabel("Predicted Total Fare (USD)")
plt.title("Regression Tree: Actual vs. Predicted")
plt.legend()
plt.show()

# Visualize the structure of the regression tree (limit to first 2 levels for clarity)
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, max_depth=2)
plt.title("Regression Tree Visualization (First 2 Levels)")
plt.show()

# Summarize evaluation metrics for both models
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Regression Tree'],
    'RMSE': [rmse_lr, rmse_tree],
    'R-squared': [r2_lr, r2_tree]
})
print("\nComparison of Models:")
print(metrics_df)



from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 9, 11],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'min_impurity_decrease': [0.0, 0.001, 0.01, 0.1]
}

# Initialize the Decision Tree Regressor
tree = DecisionTreeRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model from the grid search
best_tree = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred_best_tree = best_tree.predict(X_test)

# Evaluate the best model (using RMSE as an example)
rmse_best_tree = np.sqrt(mean_squared_error(y_test, y_pred_best_tree))
print("RMSE of best model:", rmse_best_tree)

project_folder = '/content/drive/My Drive/Colab Notebooks/'

file_path = os.path.join(project_folder, "processed_data.csv")

# Read the CSV
df = pd.read_csv(file_path)

# Define the number of bins (for quantiles) and the maximum samples per bin
num_bins = 5
n_samples_per_bin = 200

# Create a temporary binned column using pd.qcut.
# This column is only used for sampling and will be dropped later.
df['temp_fare_bin'] = pd.qcut(df['totalFare'], q=num_bins, duplicates='drop')

# For each bin, sample up to n_samples_per_bin rows (or all rows if fewer exist)
balanced_subset = (
    df.groupby('temp_fare_bin', group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), n_samples_per_bin), random_state=42))
      .reset_index(drop=True)
)

print("Balanced subset shape:", balanced_subset.shape)

# Drop the temporary binned column so that the column structure remains unchanged.
balanced_subset = balanced_subset.drop(columns=['temp_fare_bin'])

# Overwrite the original df with this balanced subset for the remainder of the analysis.
df = balanced_subset.copy()

# Display column names to verify structure
print("Columns in dataset:")
print(df.columns)

# Preview the first few rows and columns
df.head(10)

import re

def parse_duration(duration_str):
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?', duration_str)
    if not match:
        return np.nan
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes

df['travelDurationMinutes'] = df['travelDuration'].apply(parse_duration)

df['cost_per_mile'] = df['totalFare'] / df['totalTravelDistance']

# Drop rows with infinite or NaN values in the new feature
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['cost_per_mile'])

selected_features = ['travelDurationMinutes', 'totalTravelDistance', 'isBasicEconomy', 'isRefundable', 'isNonStop', 'cost_per_mile']
X = df[selected_features]
y = df['totalFare']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tree = DecisionTreeRegressor(random_state=42, **grid_search.best_params_)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
print("RMSE of tree model with cost per mile feature:", rmse_tree)











