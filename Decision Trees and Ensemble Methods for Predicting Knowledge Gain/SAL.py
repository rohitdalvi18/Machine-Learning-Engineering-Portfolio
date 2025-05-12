# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import graphviz
import random
import time
import warnings
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Ignore warnings
warnings.filterwarnings('ignore')

# mount Google Drive
from google.colab import drive
import os

#Mount Google Drive
drive.mount('/content/drive', force_remount=True)

#Specify folder to access
project_folder = '/content/drive/My Drive/Colab Notebooks/Project3SAL/'

file_path = os.path.join(project_folder, "sal_lightning_combined.csv")

df = pd.read_csv(file_path)

df.head()

df.info()

print("Shape:", df.shape)

if 'DSSQ_mean' in df.columns:
    df.drop(columns='DSSQ_mean', inplace=True)

df.info()

print("Missing values per column:\n", df.isnull().sum())

df.dropna(inplace=True)

numeric = ['session_duration','num_pages_visited','num_clicks','num_scrolls']
for col in numeric:
    lo, hi = df[col].quantile([.01, .99])
    df[col] = df[col].clip(lo, hi)

def classify_knowledge_gain(kg):
    z = (kg - kg.mean())/kg.std()
    cats = np.empty(len(z), dtype=object)
    cats[z < -0.5]               = 'Low'
    cats[(z >= -0.5) & (z <= 0.5)] = 'Moderate'
    cats[z > 0.5]                = 'High'
    return cats

df['kg_class'] = classify_knowledge_gain(df['kg_mc'])
df['kg_class'].value_counts().plot(kind='bar', title='Knowledge Gain Classes')
plt.show()

# Distribution of numeric features

df[numeric].hist(bins=20, figsize=(12,8))
plt.tight_layout()
plt.show()

# Boxplots of key behavior features by kg_class
for feat in numeric:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='kg_class', y=feat, data=df, order=['Low','Moderate','High'])
    plt.title(f"{feat} by Knowledge Gain Class")
    plt.show()

# Pairwise scatter / correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[numeric].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation of Numeric Features")
plt.show()

train_df, test_df = train_test_split(df, test_size=0.2,
                                     stratify=df['kg_class'],
                                     random_state=42)

# Scale numeric features
scaler = StandardScaler()
train_df[numeric] = scaler.fit_transform(train_df[numeric])
test_df[numeric]  = scaler.transform(test_df[numeric])

# One-hot encode categorical demographics
cat_feats = ['d_sex','d_field_of_study','d_lang']
train_df = pd.get_dummies(train_df,  columns=cat_feats, drop_first=True)
test_df  = pd.get_dummies(test_df,   columns=cat_feats, drop_first=True)

# Align train/test columns
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

print("Training set shape:", train_df.shape)
print("Test set shape:",     test_df.shape)

class Node:
    """A node in the decision tree."""
    def __init__(self, feature_idx=None, threshold=None, left=None,
                 right=None, value=None, gain=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        """Builds the tree."""
        self.n_classes_ = len(np.unique(y))
        self.tree = self._split_node(X, y, depth=0)

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self._predict_input(x, self.tree) for x in X])

    def _calculate_impurity(self, y):
        """Gini or entropy impurity of array y."""
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / m
        if self.criterion == 'gini':
            return 1 - np.sum(probs**2)
        # entropy
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _best_split(self, X, y):
        """Find feature and threshold that yields best information gain."""
        m, n = X.shape
        parent_impurity = self._calculate_impurity(y)
        best_gain, best_idx, best_thr = -1, None, None

        for idx in range(n):
            for thr in np.unique(X[:, idx]):
                left_mask = X[:, idx] <= thr
                right_mask = ~left_mask
                if (left_mask.sum() < self.min_samples_split
                        or right_mask.sum() < self.min_samples_split):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                w_imp = (len(y_left)*self._calculate_impurity(y_left) +
                         len(y_right)*self._calculate_impurity(y_right)) / m
                gain = parent_impurity - w_imp
                if gain > best_gain:
                    best_gain, best_idx, best_thr = gain, idx, thr

        return best_idx, best_thr, best_gain

    def _split_node(self, X, y, depth):
        """Recursively splits nodes until stopping criteria."""
        # Stopping criteria
        if (depth >= self.max_depth or len(np.unique(y)) == 1
                or len(y) < self.min_samples_split):
            leaf_val = self._majority_class(y)
            return Node(value=leaf_val)

        idx, thr, gain = self._best_split(X, y)
        if idx is None:
            return Node(value=self._majority_class(y))

        left_mask = X[:, idx] <= thr
        left = self._split_node(X[left_mask], y[left_mask], depth+1)
        right = self._split_node(X[~left_mask], y[~left_mask], depth+1)
        return Node(feature_idx=idx, threshold=thr, left=left, right=right, gain=gain)

    def _majority_class(self, y):
        """Return the most common class label in y."""
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _predict_input(self, x, node):
        """Traverse the tree to make a single prediction."""
        if node.value is not None:
            return node.value
        branch = node.left if x[node.feature_idx] <= node.threshold else node.right
        return self._predict_input(x, branch)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion='gini')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", dt_accuracy)

print(classification_report(y_test, y_pred, target_names=iris.target_names))

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Features and label
X_train = train_df.drop(columns=['p_id','kg_mc','kg_class']).values
X_test  = test_df.drop(columns=['p_id','kg_mc','kg_class']).values

le = LabelEncoder()
y_train = le.fit_transform(train_df['kg_class'])
y_test  = le.transform(test_df['kg_class'])

# Train & predict
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
print("Test Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=le.classes_))

df = pd.read_csv(file_path)
df.info()

print("Shape:", df.shape)

# Drop DSSQ_mean if present
if 'DSSQ_mean' in df.columns:
    df.drop(columns='DSSQ_mean', inplace=True)

df.info()

print("Missing values per column:\n", df.isnull().sum())

df.dropna(inplace=True)

numeric = ['session_duration','num_pages_visited','num_clicks','num_scrolls']
for col in numeric:
    lo, hi = df[col].quantile([.01, .99])
    df[col] = df[col].clip(lo, hi)

def classify_knowledge_gain(kg):
    z = (kg - kg.mean())/kg.std()
    cats = np.empty(len(z), dtype=object)
    cats[z < -0.5]               = 'Low'
    cats[(z >= -0.5) & (z <= 0.5)] = 'Moderate'
    cats[z > 0.5]                = 'High'
    return cats

df['kg_class'] = classify_knowledge_gain(df['kg_mc'])
df['kg_class'].value_counts().plot(kind='bar', title='Knowledge Gain Classes')
plt.show()

df[numeric].hist(bins=20, figsize=(12,8))
plt.tight_layout()
plt.show()

for feat in numeric:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='kg_class', y=feat, data=df, order=['Low','Moderate','High'])
    plt.title(f"{feat} by Knowledge Gain Class")
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation of Numeric Features")
plt.show()

# Split
train_df, test_df = train_test_split(df, test_size=0.2,
                                     stratify=df['kg_class'],
                                     random_state=42)

# Scale numeric features
scaler = StandardScaler()
train_df[numeric] = scaler.fit_transform(train_df[numeric])
test_df[numeric]  = scaler.transform(test_df[numeric])

# One-hot encode categorical demographics
cat_feats = ['d_sex','d_field_of_study','d_lang']
train_df = pd.get_dummies(train_df,  columns=cat_feats, drop_first=True)
test_df  = pd.get_dummies(test_df,   columns=cat_feats, drop_first=True)

# Align train/test columns
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

print("Training set shape:", train_df.shape)
print("Test set shape:",     test_df.shape)

X_train = train_df.drop(columns=['p_id','kg_mc','kg_class']).values
X_test  = test_df.drop(columns=['p_id','kg_mc','kg_class']).values
y_train = le.transform(train_df['kg_class'])
y_test  = le.transform(test_df['kg_class'])

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train, y_train)  [2]

# Top 10 Feature Importances
feat_names = train_df.drop(columns=['p_id','kg_mc','kg_class']).columns
importances = rf_clf.feature_importances_
imp_df = (
    pd.DataFrame({'feature': feat_names, 'importance': importances})
      .sort_values('importance', ascending=False)
      .head(10)
)

plt.figure(figsize=(6,4))
sns.barplot(x='importance', y='feature', data=imp_df, palette='viridis')
plt.title("Top 10 RF Feature Importances")
plt.tight_layout()
plt.show()

# Confusion Matrices (DT vs RF)
fig, axes = plt.subplots(1,2, figsize=(10,4), sharey=True)

# Decision Tree
cm_dt = confusion_matrix(y_test, clf.predict(X_test))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[0].set_title("Decision Tree")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Random Forest
y_pred_rf = rf_clf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
  'n_estimators': [50, 100, 200],
  'max_depth': [None, 5, 10],
  'max_features': ['sqrt', 'log2', 0.5],
  'min_samples_leaf': [1, 2, 5],
  'class_weight': ['balanced', None]
}
gs = GridSearchCV(
  RandomForestClassifier(random_state=42),
  param_grid, cv=5, scoring='f1_macro', n_jobs=-1
)
gs.fit(X_train, y_train)
print("Best RF params:", gs.best_params_)
rf_best = gs.best_estimator_

# Address Class Imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
print("Resampled dataset shape:", np.bincount(y_res))

# Retrain Random Forest with best params on resampled data
rf_balanced = RandomForestClassifier(**gs.best_params_, random_state=42)
rf_balanced.fit(X_res, y_res)

# Evaluate on original test set
y_pred_bal = rf_balanced.predict(X_test)
print("Balanced RF Accuracy:", accuracy_score(y_test, y_pred_bal))

print(classification_report(y_test, y_pred_bal, target_names=le.classes_))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bal))

!pip install scikit-learn --upgrade

import sklearn
print(sklearn.__version__)

feat_names = train_df.drop(columns=['p_id','kg_mc','kg_class']).columns

# Base AdaBoost with decision stumps
ada = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42,
)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("AdaBoost (default) Accuracy:", accuracy_score(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada, target_names=le.classes_))

from sklearn.tree import DecisionTreeClassifier

# Hyperparameter grid search for AdaBoost
param_grid_ada = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'estimator': [DecisionTreeClassifier(max_depth=1),
                      DecisionTreeClassifier(max_depth=2),
                      DecisionTreeClassifier(max_depth=3)]
}


gs_ada = GridSearchCV(
    AdaBoostClassifier(random_state=42),
    param_grid_ada,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)
gs_ada.fit(X_train, y_train)

print("Best AdaBoost params:", gs_ada.best_params_)

# Evaluate tuned AdaBoost
ada_best = gs_ada.best_estimator_
y_pred_ada_best = ada_best.predict(X_test)
print("AdaBoost (tuned) Accuracy:", accuracy_score(y_test, y_pred_ada_best))
print(classification_report(y_test, y_pred_ada_best, target_names=le.classes_))

# Base XGBoost model
xgb_clf = xgb.XGBClassifier(
    num_class=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
print("XGBoost (default) Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# Hyperparameter grid search for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
gs_xgb = GridSearchCV(
    xgb.XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid_xgb, scoring='accuracy', cv=5, n_jobs=-1
)
gs_xgb.fit(X_train, y_train)
print("Best XGBoost params:", gs_xgb.best_params_)

# Evaluate tuned XGBoost
xgb_best = gs_xgb.best_estimator_
y_pred_xgb_best = xgb_best.predict(X_test)
print("XGBoost (tuned) Accuracy:", accuracy_score(y_test, y_pred_xgb_best))
print(classification_report(y_test, y_pred_xgb_best, target_names=le.classes_))

from sklearn.metrics import f1_score

results = {
    'Model': ['DecisionTree','RandomForest','AdaBoost','AdaBoost (tuned)','XGBoost','XGBoost (tuned)'],
    'Accuracy': [
        dt_accuracy,
        rf_balanced.score(X_test, y_test),
        accuracy_score(y_test, y_pred_ada),
        accuracy_score(y_test, y_pred_ada_best),
        accuracy_score(y_test, y_pred_xgb),
        accuracy_score(y_test, y_pred_xgb_best)
    ],
    'Macro-F1': [
        f1_score(y_test, clf.predict(X_test), average='macro'),
        f1_score(y_test, rf_balanced.predict(X_test), average='macro'),
        f1_score(y_test, y_pred_ada, average='macro'),
        f1_score(y_test, y_pred_ada_best, average='macro'),
        f1_score(y_test, y_pred_xgb, average='macro'),
        f1_score(y_test, y_pred_xgb_best, average='macro')
    ]
}
pd.DataFrame(results)

