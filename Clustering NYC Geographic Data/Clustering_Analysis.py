# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SklearnKMeans, DBSCAN
from sklearn.metrics import silhouette_score
import json # used to read in NYC public art dataset

class KMeans:
    def __init__(self, k=5, max_iterations=100, random_state=None):

        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.assignments = None

    def fit(self, X):     # dataset X

        X = np.array(X)
        n_samples, n_features = X.shape

        # Initializing centroids randomly from dataset
        if self.random_state is not None:
            np.random.seed(self.random_state)
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(self.max_iterations):
            # Assign clusters based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # shape: (n_samples, k)
            new_assignments = np.argmin(distances, axis=1)

            # Check if assignments do not change
            if self.assignments is not None and np.array_equal(new_assignments, self.assignments):
                print(f"Convergence reached at iteration {iteration}")
                break

            self.assignments = new_assignments

            # Update centroids as mean of assigned points
            for i in range(self.k):
                points_in_cluster = X[self.assignments == i]
                # Avoid division by zero if a cluster gets no points
                if len(points_in_cluster) > 0:
                    self.centroids[i] = np.mean(points_in_cluster, axis=0)
                else:
                    # Reinitialize centroid to a random point if it loses all points
                    self.centroids[i] = X[np.random.choice(n_samples)]


        # Returns list of cluster assignments & list of centroid coordinates
        return self.assignments.tolist(), self.centroids.tolist()

        """
        Predict the closest cluster each sample in X belongs to.
        Requires that fit() has been called already.

        - X (array-like): n x d dataset.

        - assignments (list): Cluster assignments for each instance.
        """
    def predict(self, X):
        X = np.array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - np.array(self.centroids), axis=2)
        assignments = np.argmin(distances, axis=1)
        return assignments.tolist()


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Generate synthetic dataset
X, true_labels = make_blobs(n_samples=700, centers=4, cluster_std=0.60, random_state=0)

# Run k-Means implementation
kmeans_custom = KMeans(k=4, max_iterations=100, random_state=0)
custom_assignments, custom_centroids = kmeans_custom.fit(X)

# Run scikit-learn's k-Means implementation
kmeans_sklearn = SKLearnKMeans(n_clusters=4, random_state=0, n_init=10)
sklearn_assignments = kmeans_sklearn.fit_predict(X)
sklearn_centroids = kmeans_sklearn.cluster_centers_

# Function to plot results
def plot_clusters(X, labels, centroids, title):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=ListedColormap(['red', 'blue', 'green', 'purple']), alpha=0.6, edgecolors='k')

    # Plot centroids only if they exist
    if len(centroids) > 0:
        centroids = np.array(centroids)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', edgecolors='black', marker='X', label='Centroids')
        plt.legend()

    plt.title(title)
    plt.show()

# Plot true clusters
plot_clusters(X, true_labels, [], "True Cluster Assignments")

# Plot my k-Means results
plot_clusters(X, custom_assignments, custom_centroids, "My Custom k-Means Clustering")

# Plot scikit-learn's k-Means results
plot_clusters(X, sklearn_assignments, sklearn_centroids, "Scikit-learn k-Means Clustering")

# mount Google Drive
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Specify the folder
project_folder = "/content/drive/My Drive/Colab Notebooks/"

schools_file_path = os.path.join(project_folder, "NYC-Public-Schools-Dataset.csv")

# Load NYC Public Schools dataset
schools_df = pd.read_csv(schools_file_path)

# dimensions of the dataframe
print(schools_df.shape)

# display the top 5 rows
schools_df.head(10)

schools_df.info()

schools_df.describe()

# Select only relevant columns
schools_df = schools_df[['LATITUDE', 'LONGITUDE']].dropna()

# Convert to NumPy array for clustering
school_locations = schools_df.to_numpy()

# Sanity check: Print min/max values to ensure they are within expected NYC range
print("Latitude range:", schools_df['LATITUDE'].min(), "to", schools_df['LATITUDE'].max())
print("Longitude range:", schools_df['LONGITUDE'].min(), "to", schools_df['LONGITUDE'].max())

print(schools_df[schools_df['LONGITUDE'] == 0])

schools_df = schools_df[
    (schools_df["LATITUDE"] > 0) &
    (schools_df["LONGITUDE"] < 0)
]

print("Latitude range:", schools_df["LATITUDE"].min(), "to", schools_df["LATITUDE"].max())
print("Longitude range:", schools_df["LONGITUDE"].min(), "to", schools_df["LONGITUDE"].max())

print(schools_df.shape)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Extract coordinates
locations = schools_df[['LATITUDE', 'LONGITUDE']]

# Apply k-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
schools_df['Cluster'] = kmeans.fit_predict(locations)

# Scatter plot
plt.figure(figsize=(8,6))
plt.scatter(schools_df["LONGITUDE"], schools_df["LATITUDE"], c=schools_df["Cluster"], cmap='viridis', s=10)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,0], c='red', marker='X', s=200, label="Centroids")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("NYC Public Schools Clustering (k-Means)")
plt.legend()
plt.show()

art_file_path = os.path.join(project_folder, "NYC-Public-Art-Dataset.json")

# Load NYC Public Art dataset
with open(art_file_path) as f:
    art_df = json.load(f)

# Convert the list of dictionaries to a Pandas DataFrame
art_df = pd.DataFrame(art_df)

# dimensions of the dataframe
art_df.shape

# display the top 10 rows
art_df.head(10)

# Display initial info
art_df.info()

# Check for missing values
print(art_df[["lat", "lng"]].isna().sum())

#Convert to numeric
art_df["lat"] = pd.to_numeric(art_df["lat"], errors="coerce")
art_df["lng"] = pd.to_numeric(art_df["lng"], errors="coerce")

# Drop rows missing lat/lng
art_df = art_df.dropna(subset=["lat","lng"]).copy()
print("Shape after dropping missing coords:", art_df.shape)

# Range filter for lat/lng in NYC
in_nyc_art = (
    (art_df["lat"] >= 40.0) & (art_df["lat"] <= 41.0) &
    (art_df["lng"] >= -74.5) & (art_df["lng"] <= -73.0)
)
art_df = art_df[in_nyc_art].copy()
print("shape after range filter: ", art_df.shape)

coords_art = art_df[["lat","lng"]].values
for k in [2,3,4,5]:
    km = SklearnKMeans(n_clusters=k, random_state=42)
    labs = km.fit_predict(coords_art)
    sil = silhouette_score(coords_art, labs)
    print(f"K={k}, public art silhouette={sil:.4f}")

km_art_4 = SklearnKMeans(n_clusters=4, random_state=42)
labels_art_4 = km_art_4.fit_predict(coords_art)
art_df["cluster_kmeans"] = labels_art_4

db_art = DBSCAN(eps=0.008, min_samples=5)
db_labels_art = db_art.fit_predict(coords_art)
art_df["cluster_dbscan"] = db_labels_art

n_clusters_art_db = len(set(db_labels_art)) - (1 if -1 in db_labels_art else 0)
n_outliers_art = sum(db_labels_art == -1)

print("DBSCAN on art dataset:")
print("clusters: ", n_clusters_art_db)
print("outliers:", n_outliers_art)

# K-means
plt.scatter(art_df["lng"], art_df["lat"], c=art_df["cluster_kmeans"], cmap="tab10", alpha=0.7)
plt.title("NYC Public Art Dataset: K-means (k=5)")
plt.show()

# DBSCAN
plt.scatter(art_df["lng"], art_df["lat"], c=art_df["cluster_dbscan"], cmap="rainbow", alpha=0.7)
plt.title("NYC-Public-Art-Dataset: DBSCAN")
plt.show()

class KMeans_Extension:
    def __init__(self, k=5, max_iterations=100, balanced=False):
        """
        :param k: number of clusters
        :param max_iterations: max iterations for the algorithm
        :param balanced: if True, try to maintain balanced cluster sizes
        """
        self.k = k
        self.max_iterations = max_iterations
        self.balanced = balanced
        self.centroids = None

    def fit(self, X):
        """
        Fits the K-Means model to the data X using `k` clusters
        and up to `max_iterations` iterations. If self.balanced is True,
        tries to keep cluster sizes roughly equal.
        """
        np.random.seed(42)

        # Randomly select initial centroids from X
        initial_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[initial_indices].copy()

        for _ in range(self.max_iterations):
            # Assign clusters
            if self.balanced:
                clusters = self._assign_clusters_balanced(X)
            else:
                clusters = self._assign_clusters_default(X)

            # Compute new centroids
            new_centroids = self._compute_centroids(X, clusters)

            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        return clusters, self.centroids

    def _assign_clusters_default(self, X):
        distance_matrix = self._compute_distance_matrix(X, self.centroids)
        clusters = np.argmin(distance_matrix, axis=1)
        return clusters

    def _assign_clusters_balanced(self, X):
        """
        Balanced assignment heuristic:
        1. Compute distance to each centroid.
        2. Sort points by their best (minimum) distance to a centroid.
        3. Assign points in that order, respecting a max capacity per cluster.
        """
        n_samples = X.shape[0]
        capacity = n_samples // self.k  # integer division
        # if n_samples isn't divisible by k, a few clusters can hold 1 more.

        # 1) distance_matrix[i][j] = distance of X[i] to centroid j
        distance_matrix = self._compute_distance_matrix(X, self.centroids)

        # 2) for each point i, get the centroid preference order
        sorted_indices = np.argsort(distance_matrix, axis=1)

        # keep track of cluster assignments
        clusters = -1 * np.ones(n_samples, dtype=int)

        # keep track of how many points each cluster has
        cluster_counts = np.zeros(self.k, dtype=int)

        # sort all points by their best-dist, i.e. how sure they are of a “best centroid”
        best_dist = np.min(distance_matrix, axis=1)
        point_order = np.argsort(best_dist)

        for i in point_order:
            # go through the centroid preferences for point i
            for c_idx in sorted_indices[i]:
                # if cluster c_idx is under capacity, assign point i
                if cluster_counts[c_idx] < capacity:
                    clusters[i] = c_idx
                    cluster_counts[c_idx] += 1
                    break
            # if all clusters are at capacity, place point i in its single best centroid
            if clusters[i] == -1:
                # force it into best centroid
                c_idx = sorted_indices[i, 0]
                clusters[i] = c_idx
                cluster_counts[c_idx] += 1

        return clusters

    def _compute_centroids(self, X, clusters):
        """
        Compute new centroids by taking the mean of points assigned to each cluster.
        If no points assigned to a cluster, randomly re-initialize that centroid.
        """
        new_centroids = []
        for cluster_idx in range(self.k):
            points_in_cluster = X[clusters == cluster_idx]
            if len(points_in_cluster) > 0:
                new_centroids.append(points_in_cluster.mean(axis=0))
            else:
                new_centroids.append(X[np.random.choice(X.shape[0])])
        return np.array(new_centroids)

    def _compute_distance_matrix(self, X, centroids):
        """
        Compute Euclidean distance matrix of shape
        """
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, self.k))
        for j, c in enumerate(centroids):
            dist_to_c = np.sqrt(np.sum((X - c) ** 2, axis=1))
            distance_matrix[:, j] = dist_to_c
        return distance_matrix


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SklearnKMeans

X, true_cluster_assignments = make_blobs(
    n_samples=700,
    centers=4,
    cluster_std=0.60,
    random_state=0
)
print(X.shape)
print(np.unique(true_cluster_assignments))

#Balanced K-Means
my_kmeans_bal = KMeans_Extension(k=4, max_iterations=100, balanced=True)
pred_bal_clusters, bal_centroids = my_kmeans_bal.fit(X)


# Defauly K-Means (my implementation, balanced=False)
my_kmeans_van = KMeans_Extension(k=4, max_iterations=100, balanced=False)
pred_van_clusters, van_centroids = my_kmeans_van.fit(X)

# scikit-learn's K-Means
sk_kmeans = SklearnKMeans(n_clusters=4, max_iter=100, random_state=42, n_init=10)
sk_clusters = sk_kmeans.fit_predict(X)
sk_centroids = sk_kmeans.cluster_centers_

def plot_clusters(X, clusters, centroids, title):
    plt.scatter(X[:, 0], X[:, 1], c=clusters, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
    plt.title(title)
    plt.show()

# Plot Balanced
plot_clusters(X, pred_bal_clusters, bal_centroids, "Balanced KMeans Extensiom")

# Plot Default
plot_clusters(X, pred_van_clusters, van_centroids, "Default KMeans")

# Plot scikit-learn
plot_clusters(X, sk_clusters, sk_centroids, "scikit-learn KMeans")

#'true' cluster assignments
plot_clusters(X, true_cluster_assignments, sk_centroids, "True Cluster Assignments")

# cluster counts for balanced approach
print("Balanced approach cluster counts:", np.bincount(pred_bal_clusters))
print("Vanilla approach cluster counts:", np.bincount(pred_van_clusters))
print("sklearn approach cluster counts:", np.bincount(sk_clusters))
