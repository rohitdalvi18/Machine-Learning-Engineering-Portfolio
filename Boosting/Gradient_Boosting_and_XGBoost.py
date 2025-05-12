# install XGBoost package
!pip install xgboost

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

df = sns.load_dataset("penguins")
print("Dataset shape:", df.shape)
df.head()

# 1. Class balance
print(df['species'].value_counts())

# 2. Missing values
print(df.isnull().sum())

# 3. Feature distribution
df.hist(figsize=(10, 8))
plt.show()

# 4. Pairplot
sns.pairplot(df, hue='species')
plt.show()

# Drop missing
df2 = df.dropna().reset_index(drop=True)
print("After dropna:", df2.shape)

# # USE THIS FOR BINARY Gentoo VERSUS REST CLASSIFICATION
# # One-hot encode
# df_enc = pd.get_dummies(df2, columns=['island','sex','species'], drop_first=True)

# # Features and target
# X = df_enc.drop(columns=['species_Gentoo'])
# y = df_enc['species_Gentoo']  # 1 = Gentoo, 0 = others

# One-hot encode non-target categoricals only
df_enc = pd.get_dummies(df2, columns=['island', 'sex'], drop_first=True)

# Multiclass target: use label encoding
le = LabelEncoder()
y = le.fit_transform(df2['species'])  # Adelie=0, Chinstrap=1, Gentoo=2

X = df_enc.drop(columns=['species'])  # Keep everything else

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}
models['XGBoost'] = xgb.XGBClassifier(
    eval_metric='logloss', random_state=42
)

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    print(f"Trained {name}")

results = {}
for name, mdl in models.items():
    preds = mdl.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    results[name] = acc

plt.figure(figsize=(5,3))
plt.bar(results.keys(), results.values(), color=['C0','C1','C2'][:len(results)])
plt.ylim(0,1)
plt.title("Test Set Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

from sklearn.model_selection import GridSearchCV

# 6.3.1 RF tuning
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10],
}
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1
)
gs_rf.fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)

# 6.3.2 GB tuning
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
gs_gb = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1
)
gs_gb.fit(X_train, y_train)
print("Best GB params:", gs_gb.best_params_)

# 6.3.3 XGB tuning
param_grid_xgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
gs_xgb = GridSearchCV(
    xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1
)
gs_xgb.fit(X_train, y_train)
print("Best XGB params:", gs_xgb.best_params_)

best_rf = gs_rf.best_estimator_
best_gb = gs_gb.best_estimator_
models_tuned = {
    'RF_tuned': best_rf,
    'GB_tuned': best_gb
}

models_tuned['XGB_tuned'] = gs_xgb.best_estimator_

tuned_results = {}
for name, mdl in models_tuned.items():
    preds = mdl.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    tuned_results[name] = acc

# Bar chart of tuned models
plt.figure(figsize=(5,3))
plt.bar(tuned_results.keys(), tuned_results.values(), color=['C0','C1','C2'][:len(tuned_results)])
plt.ylim(0,1)
plt.title("Tuned Model Accuracy")
plt.ylabel("Accuracy")
plt.show()

import matplotlib.pyplot as plt

for name, mdl in models.items():
    # get feature importances and sort
    imp = mdl.feature_importances_
    idx = np.argsort(imp)[::-1][:10]
    top_feats = X_train.columns[idx]

    plt.figure(figsize=(6,4))
    plt.title(f"{name} – Top 10 Features")
    plt.bar(range(len(idx)), imp[idx])
    plt.xticks(range(len(idx)), top_feats, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

from sklearn.inspection import PartialDependenceDisplay

# identify top 2 GB features
gb = models['GradientBoosting']
imp_gb = gb.feature_importances_
gb_idx = np.argsort(imp_gb)[::-1]
top2 = [X_train.columns[i] for i in gb_idx[:2]]

fig, ax = plt.subplots(figsize=(8,5))
PartialDependenceDisplay.from_estimator(
    gb, X_train, top2,
    target=2,  # Class index for Gentoo (change for different species/classes)
    feature_names=X_train.columns, ax=ax
)
plt.tight_layout()
plt.show()

# 1) Gain‐based feature importance
fig, ax = plt.subplots(figsize=(6,4))
xgb.plot_importance(models['XGBoost'], max_num_features=10, ax=ax)
plt.title("XGBoost – Feature Importance (gain)")
plt.tight_layout()
plt.show()

# 2) Visualize tree #0
fig, ax = plt.subplots(figsize=(8,6))
xgb.plot_tree(models['XGBoost'], num_trees=0, ax=ax)
plt.title("XGBoost – Tree #0 Structure")
plt.tight_layout()
plt.show()

# Get class probabilities for each sample
xgb_model = models['XGBoost']
xgb_probs = xgb_model.predict_proba(X_test)

# Show probability predictions for first 5 samples
pd.DataFrame(xgb_probs, columns=xgb_model.classes_).head()


