!pip install datasets

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
from sklearn.linear_model import LogisticRegression

# Load dataset
# Shuffle the full training set, then take 1000 samples
dataset = load_dataset("rotten_tomatoes", split="train").shuffle(seed=42).select(range(2000))
df = pd.DataFrame(dataset)
df.head()

df['label'] = df['label'].astype(int)

sns.countplot(x='label', data=df)
plt.title('Sentiment Class Distribution')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.ylabel('Count')
plt.show()

df['label'].value_counts(normalize=True)

# Display sample reviews from each class
print("🔹 Sample Negative Reviews:")
print(df[df['label'] == 0]['text'].sample(3, random_state=1).to_string(index=False), "\n")

print("🔹 Sample Positive Reviews:")
print(df[df['label'] == 1]['text'].sample(3, random_state=2).to_string(index=False))

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label'])

# Bag-of-Words
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

# Convert sparse matrix to DataFrame (for demonstration — only do this for small samples!)
sample_sparse_df = pd.DataFrame.sparse.from_spmatrix(X_train_vec[:5], columns=vectorizer.get_feature_names_out())
print("🔍 Sparse Matrix (Bag of Words) Sample:")
display(sample_sparse_df.head())

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train_vec, y_train)
y_pred = knn.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
print(report)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')

knn.fit(X_train_vec, y_train)
y_pred = knn.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Define the range of k values to test
k_values = range(1, 21)
cv_scores = []

# Loop over k values and compute mean cross-validation accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan', weights='uniform')  # or 'distance'
    scores = cross_val_score(knn, X_train_vec, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find best k
best_k = k_values[np.argmax(cv_scores)]
print(f"✅ Best k based on CV: {best_k} (Accuracy = {max(cv_scores):.2f})")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(k_values, cv_scores, marker='o')
plt.title("Cross-Validation Accuracy vs. k (Manhattan Distance)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Mean Accuracy")
plt.grid(True)
plt.xticks(k_values)
plt.show()

# Bag of Words
bow_vectorizer = CountVectorizer(stop_words='english')
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# Cosine distance (1 - cosine similarity)
knn_cos_bow = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_cos_bow.fit(X_train_bow, y_train)
y_pred_cos_bow = knn_cos_bow.predict(X_test_bow)

print("KNN with Bag-of-Words + Cosine Distance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_cos_bow):.2f}")
print(classification_report(y_test, y_pred_cos_bow, target_names=['Negative', 'Positive']))

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

knn_manh_tfidf = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manh_tfidf.fit(X_train_tfidf, y_train)
y_pred_manh_tfidf = knn_manh_tfidf.predict(X_test_tfidf)

print("KNN with TF-IDF + Manhattan Distance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_manh_tfidf):.2f}")
print(classification_report(y_test, y_pred_manh_tfidf, target_names=['Negative', 'Positive']))

knn_cos_tfidf = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_cos_tfidf.fit(X_train_tfidf, y_train)
y_pred_cos_tfidf = knn_cos_tfidf.predict(X_test_tfidf)

print("KNN with TF-IDF + Cosine Distance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_cos_tfidf):.2f}")
print(classification_report(y_test, y_pred_cos_tfidf, target_names=['Negative', 'Positive']))

false_negatives = X_test[(y_test == 1) & (y_pred_manh_tfidf == 0)]
for i, review in enumerate(false_negatives[:5]):
    print(f"Misclassified as Negative:\n{review}\n")

correct_positives = X_test[(y_test == 1) & (y_pred_manh_tfidf == 1)]
for i, review in enumerate(correct_positives[:5]):
    print(f"Classified Correctly as Positive:\n{review}\n")

# Number of positive test examples to show
N = 3  # 👈 Increase to show more comparisons

# Get positions of positive test examples
pos_indices = np.where(y_test.values == 1)[0][:N]

# Loop over the first N positive examples
for test_idx in pos_indices:
    print("=" * 80)

    # 1. Show test review
    test_review = X_test.iloc[test_idx]
    print(f"🟢 Test Review (True Label: Positive):\n\"{test_review}\"\n")

    # 2. Get vector
    test_vec = X_test_tfidf[test_idx]

    # 3. Compute distances to training set
    manh_dists = manhattan_distances(test_vec, X_train_tfidf).flatten()
    cos_dists = cosine_distances(test_vec, X_train_tfidf).flatten()

    # 4. Get top 5 neighbor indices
    top5_manh = np.argsort(manh_dists)[:5]
    top5_cos = np.argsort(cos_dists)[:5]

    # 5. Display function
    def show_neighbors(indices, dist_array, label_array, review_array, title):
        print(f"🔍 {title} Neighbors:")
        for i in indices:
            print(f"  Distance: {dist_array[i]:.3f} | Label: {'Positive' if label_array[i] == 1 else 'Negative'}")
            print(f"  Review: \"{review_array.iloc[i]}\"\n")

    # 6. Show Manhattan and Cosine neighbors
    show_neighbors(top5_manh, manh_dists, y_train.values, X_train, "Manhattan")
    show_neighbors(top5_cos, cos_dists, y_train.values, X_train, "Cosine")


