import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Fix randomness for reproducibility
np.random.seed(42)

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

X_train.info()

y_train.describe()

# Scatter plots of features vs. target
for col in X_train.columns:
  plt.figure(figsize=(8, 6))
  plt.scatter(X_train[col], y_train, alpha=0.5)
  plt.xlabel(col)
  plt.ylabel('Target')
  plt.title(f'Scatter Plot of {col} vs. Target')
  plt.show()

# Correlation matrix
corr_matrix = X_train.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Correlation Matrix')
plt.show()

# Histograms of features
X_train.hist(bins=50, figsize=(15, 10))
plt.show()

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=100, max_depth=6, random_state=42
)
rf.fit(X_train, y_train)

# Predictions & metrics
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf  = r2_score(y_test, y_pred_rf)

print(f"RF → RMSE: {np.sqrt(mse_rf):.3f},  R²: {r2_rf:.3f}")

# Feature importances plot
importances = rf.feature_importances_
feat_names = X.columns

plt.figure(figsize=(8,4))
plt.barh(feat_names, importances)
plt.title("RF Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Train sklearn AdaBoost
ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)
mse_ada = mean_squared_error(y_test, y_pred_ada)
r2_ada  = r2_score(y_test, y_pred_ada)

print(f"AdaBoost → RMSE: {np.sqrt(mse_ada):.3f},  R²: {r2_ada:.3f}")

# Plot train/test error per iteration
train_err, test_err = [], []
for y_train_pred in ada.staged_predict(X_train):
    train_err.append(mean_squared_error(y_train, y_train_pred))
for y_test_pred in ada.staged_predict(X_test):
    test_err .append(mean_squared_error(y_test, y_test_pred))

plt.figure(figsize=(6,4))
plt.plot(np.sqrt(train_err), label="Train RMSE")
plt.plot(np.sqrt(test_err),  label="Test RMSE")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.legend()
plt.title("AdaBoost Learning Curve")
plt.tight_layout()
plt.show()

# Custom Bagged AdaBoost for regression
from sklearn.base import clone

def bagged_adaboost_regressor(
    X, y, n_estimators=50, base_learner=DecisionTreeRegressor(max_depth=1)
):
    n = len(X)
    w = np.ones(n) / n            # initialize weights
    learners, alphas = [], []

    # Normalize y-range for loss scaling
    y_arr = np.array(y)
    R = np.max(np.abs(y_arr - np.mean(y_arr)))

    for m in range(n_estimators):
        # 1) Sample with replacement ∝ w
        idx = np.random.choice(n, size=n, replace=True, p=w)
        X_samp, y_samp = X.iloc[idx], y_arr[idx]

        # 2) Fit a new learner
        clf = clone(base_learner)
        clf.fit(X_samp, y_samp)

        # 3) Predict on full set, compute normalized loss
        y_pred_full = clf.predict(X)
        loss = np.abs(y_arr - y_pred_full) / R   # in [0,1]

        # 4) Weighted error
        err = np.dot(w, loss)
        if err >= 0.5:  # skip if no improvement
            continue

        # 5) Compute alpha
        alpha = 0.5 * np.log((1 - err) / err)
        learners.append(clf);  alphas.append(alpha)

        # 6) Update weights: upweight high-loss examples
        w *= np.exp(alpha * loss)
        w /= np.sum(w)  # normalize

    return learners, alphas

def predict_bagged_ada(X, learners, alphas):
    pred = np.zeros(len(X))
    for clf, a in zip(learners, alphas):
        pred += a * clf.predict(X)
    return pred / np.sum(alphas)

# Train + evaluate
learners, alphas = bagged_adaboost_regressor(X_train, y_train, n_estimators=50)
y_pred_bag = predict_bagged_ada(X_test, learners, alphas)

mse_bag = mean_squared_error(y_test, y_pred_bag)
r2_bag  = r2_score(y_test, y_pred_bag)
print(f"Bagged AdaBoost → RMSE: {np.sqrt(mse_bag):.3f},  R²: {r2_bag:.3f}")

# Visual comparison of all three
results = pd.DataFrame({
    "Model": ["Random Forest", "AdaBoost", "Bagged AdaBoost"],
    "RMSE":   [np.sqrt(mse_rf), np.sqrt(mse_ada), np.sqrt(mse_bag)],
    "R²":     [r2_rf, r2_ada, r2_bag]
})
print(results)

# Bar chart
plt.figure(figsize=(6,4))
for i, col in enumerate(["RMSE","R²"]):
    plt.subplot(1,2,i+1)
    plt.bar(results["Model"], results[col])
    plt.xticks(rotation=15)
    plt.title(col)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y, bins=50)
plt.xlabel("Housing Price")
plt.ylabel("Frequency")
plt.title("Distribution of Housing Prices")
plt.show()

plt.figure(figsize=(8, 6))
plt.boxplot(y)
plt.ylabel("Housing Price")
plt.title("Boxplot of Housing Prices")
plt.show()

# Calculate the 95th percentile
threshold = np.percentile(y, 95)

# Filter out outliers
X_train_filtered = X_train[y_train <= threshold]
y_train_filtered = y_train[y_train <= threshold]

#Retrain the models with filtered data
#AdaBoost
ada_filtered = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada_filtered.fit(X_train_filtered, y_train_filtered)
y_pred_ada_filtered = ada_filtered.predict(X_test)
mse_ada_filtered = mean_squared_error(y_test, y_pred_ada_filtered)
r2_ada_filtered  = r2_score(y_test, y_pred_ada_filtered)
print(f"AdaBoost (filtered) → RMSE: {np.sqrt(mse_ada_filtered):.3f},  R²: {r2_ada_filtered:.3f}")

#RandomForest
rf_filtered = RandomForestRegressor(
    n_estimators=100, max_depth=6, random_state=42
)
rf_filtered.fit(X_train_filtered, y_train_filtered)
y_pred_rf_filtered = rf_filtered.predict(X_test)
mse_rf_filtered = mean_squared_error(y_test, y_pred_rf_filtered)
r2_rf_filtered  = r2_score(y_test, y_pred_rf_filtered)
print(f"RF (filtered) → RMSE: {np.sqrt(mse_rf_filtered):.3f},  R²: {r2_rf_filtered:.3f}")

# Comparison
print("\nModel Comparison Before and After Outlier Removal:")
print(f"{'Model':<15} {'RMSE (Before)':<15} {'R² (Before)':<15} {'RMSE (After)':<15} {'R² (After)':<15}")
print(f"{'AdaBoost':<15} {np.sqrt(mse_ada):<15.3f} {r2_ada:<15.3f} {np.sqrt(mse_ada_filtered):<15.3f} {r2_ada_filtered:<15.3f}")
print(f"{'RandomForest':<15} {np.sqrt(mse_rf):<15.3f} {r2_rf:<15.3f} {np.sqrt(mse_rf_filtered):<15.3f} {r2_rf_filtered:<15.3f}")



