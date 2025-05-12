import pandas as pd                         # data manipulation
import numpy as np                          # numerical operations
import matplotlib.pyplot as plt             # basic plotting
import seaborn as sns                       # statistical vizualisation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import dbscan
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score

# Generate crescent moons dataset
blobs = make_moons(500, noise=0.055, random_state=53)[0]

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler to your data and transform it
# This is mostly for practice here since the scale should be uniform
blobs_scaled = scaler.fit_transform(blobs)

# Use random module to set the state of DBSCAN since it doesn't have a built-in
# random state parameter like k-means does
import random

random.seed(53)

#Plot
plt.style.use("bmh")
fig, ax = plt.subplots(1,3, dpi=250)
fig.suptitle("k-Means versus DBSCAN Crescent Moons", fontsize=20)

# Scikit-Learn KMeans
# Setting k=2
preds_k = KMeans(2, random_state=53).fit(blobs_scaled).labels_
kmean_blob = np.append(blobs_scaled, preds_k.reshape(-1,1), axis=1)
pd.DataFrame(kmean_blob).plot(x=1, y=0, kind="scatter",ax=ax[1], c=2, colorbar=False, title= "Scikit-Learn KMeans (k=2)", marker="o", colormap="viridis")

# Scikit-Learn DBSCAN
# Setting epsilon radius to 0.2 and minimum samples to 5
preds = dbscan(blobs_scaled, eps=0.2, min_samples=5)[1]
dbscan_blob = np.append(blobs_scaled, preds.reshape(-1,1), axis=1)
pd.DataFrame(dbscan_blob).plot(x=1, y=0, kind="scatter", c=2, colorbar=False, ax=ax[2], title= "Scikit-Learn DBSCAN (eps=0.2, min_points=5)", marker="o", colormap="viridis")

# Calculate silhouette score for KMeans
kmeans_silhouette = silhouette_score(blobs_scaled, preds_k)

# Calculate silhouette score for DBSCAN
# (Only include points assigned to clusters, not noise points)
dbscan_silhouette_no_noise = silhouette_score(blobs_scaled[preds != -1], preds[preds != -1])

# Silhouette score including noise
if len(set(preds)) > 1:  # Ensure more than one cluster exists
    dbscan_silhouette_noise = silhouette_score(blobs_scaled, preds)
else:
    dbscan_silhouette_noise = -1  # Assign -1 if only one cluster exists

print(f"Silhouette Score (Excluding Noise): {dbscan_silhouette_no_noise}")
print(f"Silhouette Score (Including Noise): {dbscan_silhouette_noise}")

# Test Data
pd.DataFrame(blobs_scaled).plot(x=1, y=0, kind="scatter", ax=ax[0], alpha=0.5, figsize=(15,6), title="Test Data", marker="o", c="#e377c0")

# Update titles to include silhouette scores
ax[1].set_title(f"Scikit-Learn KMeans (k=2)\nSilhouette Score: {kmeans_silhouette:.3f}")
ax[2].set_title(f"Scikit-Learn DBSCAN (eps=0.2, min_points=5)\nSilhouette Score: {dbscan_silhouette_no_noise:.3f}")

plt.show()

# Explore different epsilon values with a fixed min_samples
epsilon_values = [0.1, 0.3, 0.6]
min_samples = 5

plt.figure(figsize=(15, 3))
for i, eps in enumerate(epsilon_values):
    labels = dbscan(blobs_scaled, eps=eps, min_samples=min_samples)[1]

    plt.subplot(1, 3, i + 1)
    core_samples_mask = labels != -1  # Core and border points
    noise_mask = labels == -1         # Noise points

    # Plot core and border points normally
    plt.scatter(blobs_scaled[core_samples_mask, 0], blobs_scaled[core_samples_mask, 1],
                c=labels[core_samples_mask], cmap='viridis', s=10)

    # Plot noise as 'X' markers
    plt.scatter(blobs_scaled[noise_mask, 0], blobs_scaled[noise_mask, 1],
                c='red', marker='x', s=20, label="Noise")

    plt.title(f"DBSCAN: ε = {eps}, min_samples = {min_samples}")
    plt.legend()

plt.tight_layout()
plt.show()

# Explore different min_samples values with a fixed epsilon
eps = 0.3
min_samples_values = [5, 20, 30]

plt.figure(figsize=(15, 4))

for i, ms in enumerate(min_samples_values):
    # Run DBSCAN with fixed eps and varying min_samples
    labels = dbscan(blobs_scaled, eps=eps, min_samples=ms)[1]

    plt.subplot(1, 3, i + 1)

    # NumPy Boolean masking:
    # labels != -1 creates a Boolean array where:
    #   - True  → Core and border points (part of a cluster)
    #   - False → Noise points (-1 label)
    core_samples_mask = labels != -1

    # labels == -1 creates a Boolean array where:
    #   - True  → Noise points (-1 label)
    #   - False → Clustered points (assigned a cluster label)
    noise_mask = labels == -1

    # Plot core and border points (normal clustered points)
    # blobs_scaled[core_samples_mask, 0] selects x-coordinates of cluster points
    # blobs_scaled[core_samples_mask, 1] selects y-coordinates of cluster points
    plt.scatter(blobs_scaled[core_samples_mask, 0], blobs_scaled[core_samples_mask, 1],
                c=labels[core_samples_mask], cmap='viridis', s=10,
                label="Clustered Points")  # Use the same colors as labels

    # Plot noise points separately in red with 'X' markers
    plt.scatter(blobs_scaled[noise_mask, 0], blobs_scaled[noise_mask, 1],
                c='red', marker='x', s=20, label="Noise")

    plt.title(f"DBSCAN: ε = {eps}, min_samples = {ms}")
    plt.legend()

plt.tight_layout()
plt.show()

# mount Google Drive
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Specify the folder you want to access
project_folder = "/content/drive/My Drive/Colab Notebooks"

file_path = os.path.join(project_folder, "Street_Tree_List_20250107.csv")

trees_df = pd.read_csv(file_path)

# Display the first few rows
trees_df.head()

import re

# Function to extract latitude and longitude from Location string
def extract_lat_lon(location_str):
    try:
        # Match pattern: (lat, lon)
        match = re.match(r"\(([-\d.]+), ([-\d.]+)\)", str(location_str))
        if match:
            return float(match.group(1)), float(match.group(2))
    except (ValueError, TypeError):
        pass
    return None, None

# Apply extraction and fill missing lat/lon values
lat_lon_data = trees_df.apply(
    lambda row: extract_lat_lon(row['Location']) if pd.isna(row['Latitude']) or pd.isna(row['Longitude']) else (row['Latitude'], row['Longitude']),
    axis=1
)

# Convert the resulting Series of tuples into two columns
trees_df['Latitude'], trees_df['Longitude'] = zip(*lat_lon_data)

# Remove rows with missing or invalid coordinates
trees_clean = trees_df.dropna(subset=['Latitude', 'Longitude'])
trees_clean = trees_clean[(trees_clean['Latitude'].between(37.5, 37.9)) &
                          (trees_clean['Longitude'].between(-123, -122))]

# Verify the cleaned data
trees_clean[['Latitude', 'Longitude']].describe()

# Plot tree locations
plt.figure(figsize=(10, 6))
plt.scatter(trees_clean['Longitude'], trees_clean['Latitude'], s=5, color='green')
plt.title('San Francisco Street Tree Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Plot tree locations
plt.figure(figsize=(8,20))
plt.scatter(trees_clean['Longitude'], trees_clean['Latitude'], s=5, color='green')
plt.title('San Francisco Street Tree Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Apply K-means clustering
kmeans = KMeans(n_clusters=11, random_state=42)
trees_clean['kmeans_cluster'] = kmeans.fit_predict(trees_clean[['Longitude', 'Latitude']])

# Plot K-means clusters
plt.figure(figsize=(8, 20))
plt.scatter(trees_clean['Longitude'], trees_clean['Latitude'], c=trees_clean['kmeans_cluster'], cmap='viridis', s=5)
plt.title('K-means Clustering of SF Trees')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()



# Define progressively smaller epsilon values
epsilon_values = [0.005, 0.001, 0.0005]  # Decreasing ε
min_samples = 30

# Prepare subplots
fig, axes = plt.subplots(1, len(epsilon_values), figsize=(25, 20), sharex=True, sharey=True)

for i, eps in enumerate(epsilon_values):
    # Apply DBSCAN clustering
    clusters = dbscan(X=trees_clean[['Longitude', 'Latitude']], eps=eps, min_samples=min_samples)
    labels = clusters[1]  # Extract cluster labels

    # Identify noise points (labeled as -1) and clusters
    noise_mask = labels == -1
    cluster_mask = labels != -1

    # Count the number of unique clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(noise_mask)

    # Plot cluster points
    scatter = axes[i].scatter(trees_clean.loc[cluster_mask, 'Longitude'],
                              trees_clean.loc[cluster_mask, 'Latitude'],
                              c=labels[cluster_mask], cmap='tab10', s=5, label='Clusters')

    # Plot noise points separately in red with "X" markers
    axes[i].scatter(trees_clean.loc[noise_mask, 'Longitude'],
                    trees_clean.loc[noise_mask, 'Latitude'],
                    c='red', marker='x', s=15, label='Noise')

    # Add title with cluster and noise count
    axes[i].set_title(f"DBSCAN: ε = {eps:.4f}\nClusters: {num_clusters}, Noise: {num_noise}")

    # Add legend to first plot only
    if i == 0:
        axes[i].legend(loc='upper right')

# Adjust layout and show the plot
plt.suptitle("DBSCAN: Effect of Varying ε on Noise and Clusters")
plt.show()



# Define progressively smaller epsilon values
epsilon_values = 0.005
min_samples = [5, 10, 30]

# Prepare subplots
fig, axes = plt.subplots(1, len(min_samples), figsize=(25, 20), sharex=True, sharey=True)

for i, ms in enumerate(min_samples):
    # Apply DBSCAN clustering
    clusters = dbscan(X=trees_clean[['Longitude', 'Latitude']], eps=eps, min_samples=ms)
    labels = clusters[1]  # Extract cluster labels

    # Identify noise points (labeled as -1) and clusters
    noise_mask = labels == -1
    cluster_mask = labels != -1

    # Count the number of unique clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = np.sum(noise_mask)

    # Plot cluster points
    scatter = axes[i].scatter(trees_clean.loc[cluster_mask, 'Longitude'],
                              trees_clean.loc[cluster_mask, 'Latitude'],
                              c=labels[cluster_mask], cmap='tab10', s=5, label='Clusters')

    # Plot noise points separately in red with "X" markers
    axes[i].scatter(trees_clean.loc[noise_mask, 'Longitude'],
                    trees_clean.loc[noise_mask, 'Latitude'],
                    c='red', marker='x', s=15, label='Noise')

    # Add title with cluster and noise count
    axes[i].set_title(f"DBSCAN: min_samples = {ms:.4f}\nClusters: {num_clusters}, Noise: {num_noise}")

    # Add legend to first plot only
    if i == 0:
        axes[i].legend(loc='upper right')

# Adjust layout and show the plot
plt.suptitle("DBSCAN: Effect of Varying min_samples on Noise and Clusters")
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

def generate_crescent_moons():
    """
    Generate crescent moon dataset with noise.
    """
    X, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
    X = StandardScaler().fit_transform(X)  # Normalize data for better clustering
    return X

def custom_dbscan(X, eps=0.3, min_samples=30, process_order=None):
    """
    Custom DBSCAN where order of processing affects border point assignment.
    """
    n_points = X.shape[0]
    if process_order is None:
        process_order = np.arange(n_points)

    labels = np.full(n_points, -1)  # Initialize all as unassigned
    core_points = np.zeros(n_points, dtype=bool)

    # Compute neighborhoods
    nbrs = NearestNeighbors(radius=eps).fit(X)
    neighborhoods = nbrs.radius_neighbors(X, eps, return_distance=False)

    # Identify core points
    for i in range(n_points):
        if len(neighborhoods[i]) >= min_samples:
            core_points[i] = True

    # Assign clusters following process_order
    current_cluster = 0
    for idx in process_order:
        if labels[idx] != -1:  # Skip already labeled points
            continue

        if not core_points[idx]:  # Skip non-core points for now
            continue

        labels[idx] = current_cluster
        stack = [idx]

        while stack:
            current_point = stack.pop()
            neighbors = neighborhoods[current_point]

            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = current_cluster
                    if core_points[neighbor]:
                        stack.append(neighbor)

        current_cluster += 1

    return labels, ~core_points & (labels != -1)

def plot_clusters(X, labels, border_points_mask, title, ax):
    """
    Visualizes clusters and highlights border points with dark grey outlines.
    """
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Assign colors dynamically

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax.scatter(X[mask, 0], X[mask, 1], c='red',
                       marker='x', s=30, label='Noise')
        else:
            core_mask = mask & ~border_points_mask
            border_mask = mask & border_points_mask

            # Core points: Normal scatter
            ax.scatter(X[core_mask, 0], X[core_mask, 1],
                       color=colors[i], s=10, alpha=0.8, label=f'Cluster {label}')

            # Border points: Same color but outlined in dark grey
            if np.any(border_mask):
                ax.scatter(X[border_mask, 0], X[border_mask, 1],
                           facecolors=colors[i], edgecolors='#333333',
                           linewidth=1, alpha=0.9, s=10, label="Border Points")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

# Generate crescent moons dataset
X = generate_crescent_moons()

# Define two different processing orders
order1 = np.argsort(X[:, 0])  # Left to right
order2 = np.argsort(-X[:, 0]) # Right to left

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Process with order 1
labels1, border_points1 = custom_dbscan(X, eps=0.3, min_samples=30, process_order=order1)
plot_clusters(X, labels1, border_points1, "Process Left-to-Right", ax1)

# Process with order 2
labels2, border_points2 = custom_dbscan(X, eps=0.3, min_samples=30, process_order=order2)
plot_clusters(X, labels2, border_points2, "Process Right-to-Left", ax2)

plt.tight_layout()
plt.show()



from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
X_housing = california.data[:, [6, 7]]  # Latitude and Longitude
y_housing = california.target  # Median house value

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_housing[:, 1], X_housing[:, 0],
                     c=y_housing, cmap='viridis', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('California Housing Prices by Location')
plt.colorbar(scatter, label='Median House Value')
plt.show()

from sklearn.cluster import DBSCAN
# Define Grid Search Parameters
eps_values = np.linspace(0.01, 0.5, 10)  # Small to large neighborhood sizes
min_samples_values = [3, 5, 10, 20]  # Varying density thresholds

# Store results
results = []

# Grid Search: Compute Silhouette Scores
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_housing)

        # Ignore single-cluster cases to avoid silhouette score errors
        if len(set(labels)) > 1:
            score = silhouette_score(X_housing, labels)
        else:
            score = -1  # Indicates poor clustering

        results.append((eps, min_samples, score))

# # Convert results to DataFrame for better visualization
#results_df = pd.DataFrame(results, columns=["Epsilon", "Min Samples", "Silhouette Score"])

# results_df

# # Display results
#import ace_tools as tools
#tools.display_dataframe_to_user(name="DBSCAN Silhouette Score Results", dataframe=results_df)

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results, columns=["Epsilon", "Min Samples", "Silhouette Score"])

results_df

fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize=(15, 15))
fig.suptitle("DBSCAN Clusters for Different Epsilon and Min Samples", fontsize=14)

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_housing)

        ax = axes[i, j]
        core_samples_mask = labels != -1  # True for clustered points
        noise_mask = labels == -1  # True for noise points

        # Scatter plot: Clustered points
        ax.scatter(X_housing[core_samples_mask, 1], X_housing[core_samples_mask, 0],
                   c=labels[core_samples_mask], cmap="viridis", s=10, alpha=0.8)

        # Scatter plot: Noise points
        ax.scatter(X_housing[noise_mask, 1], X_housing[noise_mask, 0],
                   c='red', marker='x', s=20, label="Noise")

        ax.set_title(f"ε={eps:.2f}, min_samples={min_samples}")

plt.tight_layout()
plt.show()

# Function to evaluate DBSCAN parameters
def evaluate_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    # Count number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Calculate silhouette score if more than one cluster and not all noise
    if n_clusters > 1 and n_noise < len(labels):
        score = silhouette_score(X, labels)
    else:
        score = -1

    return labels, n_clusters, n_noise, score

# Grid search over eps and min_samples
eps_range = np.arange(0.2, 1.1, 0.2)
min_samples_range = [5, 10, 15, 20]

results = []
for eps in eps_range:
    for min_samples in min_samples_range:
        labels, n_clusters, n_noise, score = evaluate_dbscan(X_housing, eps, min_samples)
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': score
        })

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(results)

# Plot parameter evaluation results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Number of clusters
for i, min_samples in enumerate(min_samples_range):
    mask = results_df['min_samples'] == min_samples
    axes[0, 0].plot(results_df[mask]['eps'],
                    results_df[mask]['n_clusters'],
                    'o-', label=f'min_samples={min_samples}')
axes[0, 0].set_xlabel('eps')
axes[0, 0].set_ylabel('Number of Clusters')
axes[0, 0].legend()

# Number of noise points
for i, min_samples in enumerate(min_samples_range):
    mask = results_df['min_samples'] == min_samples
    axes[0, 1].plot(results_df[mask]['eps'],
                    results_df[mask]['n_noise'],
                    'o-', label=f'min_samples={min_samples}')
axes[0, 1].set_xlabel('eps')
axes[0, 1].set_ylabel('Number of Noise Points')
axes[0, 1].legend()

# Silhouette score
for i, min_samples in enumerate(min_samples_range):
    mask = results_df['min_samples'] == min_samples
    axes[1, 0].plot(results_df[mask]['eps'],
                    results_df[mask]['silhouette'],
                    'o-', label=f'min_samples={min_samples}')
axes[1, 0].set_xlabel('eps')
axes[1, 0].set_ylabel('Silhouette Score')
axes[1, 0].legend()

plt.tight_layout()
plt.show()

# Apply DBSCAN with optimal parameters
best_result = results_df.loc[results_df['silhouette'].idxmax()]
optimal_eps = best_result['eps']
optimal_min_samples = int(best_result['min_samples'])

print(f'Optimal Epsilon Extracted: {optimal_eps}')
print(f'Optimal Minimum Samples Extracted: {optimal_min_samples}')

final_dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
cluster_labels = final_dbscan.fit_predict(X_housing)

# Visualize final clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_housing[:, 1], X_housing[:, 0],
                     c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clusters of California Housing')
plt.colorbar(scatter)
plt.show()



# Apply DBSCAN with your parameters

your_dbscan = DBSCAN(eps=0.34, min_samples=20)
cluster_labels = your_dbscan.fit_predict(X_housing)

# Visualize your clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_housing[:, 1], X_housing[:, 0],
                     c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DBSCAN Clusters of California Housing')
plt.colorbar(scatter)
plt.show()


