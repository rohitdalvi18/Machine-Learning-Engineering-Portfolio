import pandas as pd                         # data manipulation
import numpy as np                          # numerical operations
import matplotlib.pyplot as plt             # basic plotting
import seaborn as sns                       # statistical vizualisation
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import dbscan
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

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

# Test Data
pd.DataFrame(blobs_scaled).plot(x=1, y=0, kind="scatter", ax=ax[0], alpha=0.5, figsize=(15,6), title="Test Data", marker="o", c="#e377c0")

plt.show()

# Explore different epsilon values with a fixed min_samples
epsilon_values = [0.1, 0.3, 0.6]
min_samples = 5

plt.figure(figsize=(15, 3))
for i, eps in enumerate(epsilon_values):
    labels = dbscan(blobs_scaled, eps=eps, min_samples=min_samples)[1]
    plt.subplot(1, 3, i+1)
    plt.scatter(blobs_scaled[:, 0], blobs_scaled[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f"DBSCAN: ε = {eps}, min_samples = {min_samples}")

plt.tight_layout()
plt.show()

# Explore different min_samples values with a fixed epsilon
eps = 0.3
min_samples_values = [5, 20, 30]

plt.figure(figsize=(15, 4))
for i, ms in enumerate(min_samples_values):
    labels = dbscan(blobs_scaled, eps=eps, min_samples=ms)[1]
    plt.subplot(1, 3, i+1)
    plt.scatter(blobs_scaled[:, 0], blobs_scaled[:, 1], c=labels, cmap='plasma', s=5)
    plt.title(f"DBSCAN: ε = {eps}, min_samples = {ms}")

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
kmeans = KMeans(n_clusters=4, random_state=42)
trees_clean['kmeans_cluster'] = kmeans.fit_predict(trees_clean[['Longitude', 'Latitude']])

# Plot K-means clusters
plt.figure(figsize=(8, 20))
plt.scatter(trees_clean['Longitude'], trees_clean['Latitude'], c=trees_clean['kmeans_cluster'], cmap='viridis', s=5)
plt.title('K-means Clustering of SF Trees')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

from sklearn.cluster import dbscan

# Apply DBSCAN with initial parameters
# Pass the data matrix as the first argument
clusters = dbscan(X=trees_clean[['Longitude', 'Latitude']],
                 eps=0.005,
                 min_samples=10)  # 0.005 ~500m

# The dbscan function returns a tuple - the second element is the cluster labels
trees_clean['dbscan_cluster'] = clusters[1]

# Plot DBSCAN clusters
plt.figure(figsize=(8, 20))
plt.scatter(trees_clean['Longitude'], trees_clean['Latitude'],
           c=trees_clean['dbscan_cluster'], cmap='tab20', s=5)
plt.title('DBSCAN Clustering of SF Trees')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

from sklearn.cluster import dbscan

# Apply DBSCAN with initial parameters
# Pass the data matrix as the first argument
clusters = dbscan(X=trees_clean[['Longitude', 'Latitude']],
                 eps=0.0005,
                 min_samples=10)  # 0.005 ~500m

# The dbscan function returns a tuple - the second element is the cluster labels
trees_clean['dbscan_cluster'] = clusters[1]

# Plot DBSCAN clusters
plt.figure(figsize=(8, 20))
plt.scatter(trees_clean['Longitude'], trees_clean['Latitude'],
           c=trees_clean['dbscan_cluster'], cmap='tab20', s=5)
plt.title('DBSCAN Clustering of SF Trees')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


