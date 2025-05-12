import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# URL of the raw student-mat.csv file
url = 'https://raw.githubusercontent.com/KunjalJethwani/StudentPerformance/main/student-por.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(url, sep=';')


# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

print("Dataset shape:", df.shape)
df.head()

# Create binary target column
df['Pass'] = (df['G3'] >= 10).astype(int)

# Target distribution
print(df['Pass'].value_counts())
print(df['Pass'].value_counts(normalize=True) * 100)

# Compare average study time and absences between passed vs failed
print(df.groupby('Pass')['studytime'].mean())
print(df.groupby('Pass')['absences'].mean())

numeric_features = df.select_dtypes(include=np.number)

correlation_matrix = numeric_features.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.show()

# Drop the grade columns we aren't using as features
X = df.drop(columns=['G1', 'G2', 'G3', 'Pass'])  # feature set
y = df['Pass']  # target variable
print("Features remaining:", X.columns.tolist())


# Map binary categorical features
binary_mappings = {
    'school': {'GP': 0, 'MS': 1},
    'sex': {'M': 0, 'F': 1},
    'address': {'U': 1, 'R': 0},
    'famsize': {'LE3': 0, 'GT3': 1},  # LE3 = family size <=3, GT3 = >3
    'Pstatus': {'T': 1, 'A': 0},      # T = together (parents), A = apart
    'schoolsup': {'yes': 1, 'no': 0},
    'famsup': {'yes': 1, 'no': 0},
    'paid': {'yes': 1, 'no': 0},
    'activities': {'yes': 1, 'no': 0},
    'nursery': {'yes': 1, 'no': 0},
    'higher': {'yes': 1, 'no': 0},
    'internet': {'yes': 1, 'no': 0},
    'romantic': {'yes': 1, 'no': 0}
}


# Apply the mappings
for col, mapping in binary_mappings.items():
    X[col] = X[col].map(mapping)

# Verify a couple of columns were encoded (e.g., sex, schoolsup)
print(X[['sex', 'schoolsup']].head(5))

# One-hot encode multi-category features
X = pd.get_dummies(X, columns=['Mjob', 'Fjob', 'reason', 'guardian'], drop_first=True)
print("Features after one-hot encoding:", X.columns.tolist())
print("Total number of features now:", X.shape[1])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape[0], "Test size:", X_test.shape[0])



# Initialize a scaler and fit on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform both training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier

# Instantiate KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train (fit) the classifier on the training data
knn.fit(X_train_scaled, y_train)

# Evaluate on the training set and test set
train_acc = knn.score(X_train_scaled, y_train)
test_acc = knn.score(X_test_scaled, y_test)
print(f"K=5 Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}")

from sklearn.model_selection import cross_val_score

# Try K from 1 to 15 and record cross-validation accuracy (using 5-fold CV)
cv_scores = []
neighbors = range(1, 16)
for k in neighbors:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    # 5-fold cross-validation on training data
    scores = cross_val_score(knn_k, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Print the scores
for k, acc in zip(neighbors, cv_scores):
    print(f"K={k}, Mean CV Accuracy={acc:.3f}")


plt.plot(neighbors, cv_scores, marker='o')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Cross-Val Accuracy')
plt.title('5-fold CV Accuracy vs K')
plt.xticks(range(1,16))
plt.show()


# Retrain KNN with the best K (here assuming 5 was best)
best_k = 10
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# Evaluate on test set
y_pred = knn_best.predict(X_test_scaled)
test_acc = knn_best.score(X_test_scaled, y_test)
print(f"Best K = {best_k}, Test Accuracy = {test_acc:.3f}")

from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report (precision, recall, f1)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the positive class (Pass = 1)
y_probs = knn_best.predict_proba(X_test_scaled)[:, 1]  # column 1 is prob for class 1

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =============================
# Define Custom KNN Classifier
# =============================

class KNNClassifierScratch:
    def __init__(self, k=5):
        """
        Initialize the custom KNN classifier.

        Parameters:
            k (int): Number of nearest neighbors to consider.
        """
        self.k = k
        self.X_train = None  # Placeholder for training feature data
        self.y_train = None  # Placeholder for training labels

    def fit(self, X, y):
        """
        Fit the model using the training data.

        Since KNN is a lazy learner, fitting simply stores the data.

        Parameters:
            X (numpy.ndarray): 2D array of training data (n_samples x n_features).
            y (numpy.ndarray): 1D array of training labels.
        """
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, a, b):
        """
        Compute the Euclidean distance between two vectors a and b.

        Parameters:
            a (numpy.ndarray): 1D array representing one point.
            b (numpy.ndarray): 1D array representing another point.

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, X):
        """
        Predict the class labels for the input samples X.

        For each test sample, calculate its distance to all training samples,
        select the k closest ones, and choose the most frequent label.

        Parameters:
            X (numpy.ndarray): 2D array of test samples (n_test_samples x n_features).

        Returns:
            numpy.ndarray: 1D array of predicted class labels.
        """
        predictions = []  # To store predictions for each test sample

        # Loop over each test point
        for test_point in X:
            # Compute distances between test_point and every training point
            distances = np.array([self._euclidean_distance(test_point, x_train)
                                  for x_train in self.X_train])

            # Get the indices of the k smallest distances
            nearest_indices = np.argsort(distances)[:self.k]

            # Retrieve the labels of these nearest neighbors
            nearest_labels = self.y_train[nearest_indices]

            # Use majority vote to determine the predicted label
            predicted_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(predicted_label)

        return np.array(predictions)

# =============================
# Generate Synthetic Data
# =============================

# Create a synthetic binary classification dataset
X, y = make_classification(n_samples=200, n_features=5, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.0, random_state=42)

# Split the dataset into 80% training and 20% testing (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# =============================
# Train and Evaluate Custom KNN
# =============================

# Instantiate our custom KNN with k=5
knn_custom = KNNClassifierScratch(k=5)
# Fit the model (simply stores the training data)
knn_custom.fit(X_train, y_train)
# Predict class labels on the test set
predictions_custom = knn_custom.predict(X_test)
# Compute accuracy of the custom KNN
accuracy_custom = accuracy_score(y_test, predictions_custom)

# =============================
# Train and Evaluate sklearn KNN
# =============================

# Instantiate scikit-learn's KNeighborsClassifier with k=5
knn_sklearn = KNeighborsClassifier(n_neighbors=5)
# Fit the model using training data
knn_sklearn.fit(X_train, y_train)
# Predict class labels on the test set
predictions_sklearn = knn_sklearn.predict(X_test)
# Compute accuracy of the sklearn KNN
accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

# =============================
# Compare the Predictions & Accuracy
# =============================

print("Custom KNN predictions:")
print(predictions_custom)
print("sklearn KNN predictions:")
print(predictions_sklearn)
print()
print(f"Custom KNN Accuracy: {accuracy_custom * 100:.2f}%")
print(f"sklearn KNN Accuracy: {accuracy_sklearn * 100:.2f}%")


