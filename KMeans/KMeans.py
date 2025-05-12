import pandas as pd                         # data manipulation
import numpy as np                          # numerical operations
import matplotlib.pyplot as plt             # basic plotting
import seaborn as sns                       # statistical vizualisation
from sklearn.preprocessing import StandardScaler

# mount Google Drive
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Specify the folder you want to access
project_folder = "/content/drive/My Drive/Colab Notebooks/"

file_path = os.path.join(project_folder, "weight-height.csv")

df = pd.read_csv(file_path)

# dimensions of the dataframe
print(df.shape)

# display the top 5 rows
df.head(10)

df.info()

df.describe()

# Generate scatter plot of height and weight
plt.scatter(df.Height, df.Weight, s=1, alpha=0.1)
plt.xlabel('Height in inches')
plt.ylabel('Weight in lbs')
plt.title('Height/Weight data')
plt.show()

# Select only numeric columns for correlation calculation
num_df = df.select_dtypes(include=np.number)

# Let's check correlation
num_df.corr()

# plot the correlations
sns.heatmap(num_df.corr(method='pearson'), annot=True, cmap='coolwarm')
plt.show()

for gender in df.Gender.unique():
    g_df = df[df.Gender == gender]
    plt.scatter(g_df.Height, g_df.Weight, s=1, alpha=0.1, label=gender)
plt.legend(markerscale=10)
plt.xlabel('Height in inches')
plt.ylabel('Weight in lbs')
plt.title('Weight/Height data')
plt.show()

plt.hist(df.Height, bins=100)
plt.xlabel('Height')
plt.show()

plt.hist(df.Weight, bins=100)
plt.xlabel('Weight')
plt.show()

for gender in df.Gender.unique():
    g_df = df[df.Gender == gender]
    plt.hist(g_df.Weight, bins=100, alpha=0.5, label=gender)
plt.xlabel('Weight')
plt.legend()
plt.title('Weight distribution by gender')
plt.show()

for gender in df.Gender.unique():
    g_df = df[df.Gender == gender]
    plt.hist(g_df.Height, bins=100, alpha=0.5, label=gender)
plt.xlabel('Height')
plt.legend()
plt.title('Height distribution by gender')
plt.show()

# Select features for clustering (Height and Weight)
features = ['Height', 'Weight']

# Standardize Height and Weight
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Convert scaled features back to a DataFrame and retain Gender for visualization
scaled_df = pd.DataFrame(scaled_features, columns=features)
scaled_df['Gender'] = df['Gender'].values  # Add Gender back to the DataFrame for labeling

# Display the updated DataFrame
scaled_df.head()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
kmeans.fit(scaled_df[['Height', 'Weight']])


# `.fit()` trains the KMeans algorithm on the dataset.
# `df.iloc[:, 1:]` selects all rows and all columns starting from the second column in the DataFrame `df`.
# `iloc` is used for index-based selection.
# `[:, 1:]` means we're excluding the first column (which is gender that isn't useful for clustering).
# We make sure the data passed to KMeans contains only numeric values because KMeans is a distance-based algorithm.


kmeans.labels_

#`.labels_` returns the cluster assignments for each data point in the dataset.
# Each data point is assigned a cluster label (0, 1, 2, or 3 in this case because we specified `n_clusters=4`).
# Example: If `kmeans.labels_` returns `[0, 1, 1, 2, 0, 3, ...]`, this means:
# The first data point belongs to Cluster 0.
# The second and third data points belong to Cluster 1.
# The fourth data point belongs to Cluster 2, and so on.


kmeans.cluster_centers_

# Assign cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Reverse the scaling for cluster centers to match original data scale
centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot the data points colored by their cluster assignments
plt.figure(figsize=(10, 6))

# Plot data points using original Height and Weight, colored by cluster assignments
plt.scatter(df['Height'], df['Weight'], c=df['Cluster'], s=10, alpha=0.5)

# Plot cluster centers in the original scale
plt.scatter(centers_original_scale[:, 0], centers_original_scale[:, 1],
            s=100, color='#333333', marker='X', label='Centroids')

# Add plot labels and title
plt.legend()
plt.xlabel('Height in inches')
plt.ylabel('Weight in lbs')
plt.title('Weight/Height Data Colored by KMeans Clusters')

plt.show()

# Reverse the scaling of cluster centers to match original Height and Weight
centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

# Plotting the data with gender labels and corrected cluster centers
plt.figure(figsize=(10, 6))

# Use the same colormap (e.g., 'viridis') for gender
colors = {'Male': 'purple', 'Female': 'green'}  # Adjust these as needed

# Plot data points colored by Gender
for gender in df.Gender.unique():
    g_df = df[df.Gender == gender]
    plt.scatter(g_df.Height, g_df.Weight, s=10, alpha=0.5, label=gender, color=colors[gender])

# Plot cluster centers (now scaled back to match original data)
plt.scatter(centers_original_scale[:, 0], centers_original_scale[:, 1],
            s=100, color='#333333', marker='X', label='Centroids')

# Add plot labels and title
plt.legend(markerscale=1)
plt.xlabel('Height in inches')
plt.ylabel('Weight in lbs')
plt.title('Weight/Height Data Colored by Gender with KMeans Centroids')

plt.show()

kmeans.inertia_

# Iterate over a range of k values (number of clusters) and compute inertia for each k
kminertia = pd.DataFrame(data=[], index=range(2, 21), columns=['inertia'])

for clusters in range(2,21):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df.iloc[:,1:])
    kminertia.loc[clusters] = kmeans.inertia_

kminertia

# Let's import this to force integers on the x-axis (otherwise it gives weird decimal half values)
from matplotlib.ticker import MaxNLocator

ax = plt.gca()

kminertia.plot(kind='line', y='inertia', ax=ax, marker='o')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
# Force x-axis to display whole numbers only
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

kmeans = KMeans(n_clusters=6)
kmeans.fit(scaled_df[['Height', 'Weight']])
kmeans.labels_

# Assign cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Reverse the scaling for cluster centers to match original data scale
centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot the data points colored by their cluster assignments
plt.figure(figsize=(10, 6))

# Plot data points using original Height and Weight, colored by cluster assignments
plt.scatter(df['Height'], df['Weight'], c=df['Cluster'], s=10, alpha=0.5)

# Plot cluster centers in the original scale
plt.scatter(centers_original_scale[:, 0], centers_original_scale[:, 1],
            s=100, color='#333333', marker='X', label='Centroids')

# Add plot labels and title
plt.legend()
plt.xlabel('Height in inches')
plt.ylabel('Weight in lbs')
plt.title('Weight/Height Data Colored by KMeans Clusters')

plt.show()

kmeans.cluster_centers_

kmeans.labels_

from sklearn.metrics import silhouette_score

# Assuming kmeans has already been fitted with 6 clusters on weight/height data
labels = kmeans.labels_  # Cluster labels assigned by KMeans
silhouette_avg = silhouette_score(df[['Weight', 'Height']], labels)

print(f"Silhouette Score for 6 Clusters: {silhouette_avg:.4f}")

# Evaluate Silhouette Scores for a range of clusters (2 to 10)
silhouette_scores = {}

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[['Weight', 'Height']])
    score = silhouette_score(df[['Weight', 'Height']], kmeans.labels_)
    silhouette_scores[k] = score

# Display silhouette scores
for k, score in silhouette_scores.items():
    print(f"Silhouette Score for {k} clusters: {score:.4f}")

file_path = os.path.join(project_folder, "Mall_Customers.csv")

df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Plot histograms for Age, Annual Income, and Spending Score
df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].hist(bins=20, figsize=(12, 6))
plt.suptitle('Histograms of Age, Annual Income, and Spending Score')
plt.show()

# One-hot encode 'Gender' with 1 for Female and 0 for Male
df['Gender'] = df['Gender'].map({'Female': 1, 'Male': 0})

# Verify encoding
df[['Gender']].head()

# Select features for clustering
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Convert scaled features back to a DataFrame for easier interpretation
scaled_df = pd.DataFrame(scaled_features, columns=features)
scaled_df.head()

inertia = []
silhouette_scores = {}
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)
    silhouette_scores[k] = silhouette_score(scaled_features, kmeans.labels_)

# Plot elbow of inertia values against clusters
plt.figure(figsize=(6, 4))
plt.plot(K, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')

# Print results
for k, score in silhouette_scores.items():
    print(f"Silhouette Score for {k} clusters: {score:.4f}")


# Assuming optimal k is determined as 6
optimal_k = 6
# random_state ensures the same values re-running the code
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Reverse scaling to interpret cluster centers in original units
# This reverses the standardization process, converting the cluster centers back to their original scale
# (e.g., actual Age, Annual Income, Spending Score, and Gender values) for easier interpretation.
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
cluster_centers_df

from sklearn.decomposition import PCA

# Apply PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Add PCA components to DataFrame
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

# Plotting the clusters in PCA space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=50)
plt.title('PCA Visualization of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# PCA Loadings
# Remember, the loadings help us with interpretation and understanding the
# influence of certain variables on compnents
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)
loadings

# Use only 2 key features
X_selected = scaled_df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Define number of clusters
k = 6

# Apply standard K-Means with 2 features
kmeans = KMeans(n_clusters=k, random_state=42)
scaled_df['Cluster_KMeans'] = kmeans.fit_predict(X_selected)

# Visualize cluster sizes
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(x="Cluster_KMeans", data=scaled_df, hue="Cluster_KMeans", palette="viridis", legend=False)
plt.title("Standard K-Means Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()

# Install k-means-constrained if not installed
!pip install k-means-constrained

from k_means_constrained import KMeansConstrained

# Apply Balanced K-Means with 2 key features
balanced_kmeans = KMeansConstrained(n_clusters=k, size_min=int(len(X_selected) / k), size_max=int(len(X_selected) / k) + 1, random_state=42)
scaled_df['Cluster_BalancedKMeans'] = balanced_kmeans.fit_predict(X_selected)

# Visualize balanced cluster sizes
plt.figure(figsize=(6,4))
sns.countplot(x="Cluster_BalancedKMeans", data=scaled_df, hue="Cluster_BalancedKMeans", palette="viridis", legend=False)
plt.title("Balanced K-Means Cluster Sizes")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(x="Cluster_KMeans", data=scaled_df, ax=ax[0], hue="Cluster_KMeans", palette="viridis", legend=False)
ax[0].set_title("Standard K-Means Cluster Sizes")
ax[0].set_xlabel("Cluster")
ax[0].set_ylabel("Number of Customers")

sns.countplot(x="Cluster_BalancedKMeans", data=scaled_df, ax=ax[1], hue="Cluster_BalancedKMeans", palette="viridis", legend=False)
ax[1].set_title("Balanced K-Means Cluster Sizes")
ax[1].set_xlabel("Cluster")
ax[1].set_ylabel("Number of Customers")

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Standard K-Means plot
sns.scatterplot(x=scaled_df['Annual Income (k$)'], y=scaled_df['Spending Score (1-100)'],
                hue=scaled_df['Cluster_KMeans'], palette='viridis', ax=ax[0], legend=False)
ax[0].set_title("Standard K-Means Clustering")

# Balanced K-Means plot
sns.scatterplot(x=scaled_df['Annual Income (k$)'], y=scaled_df['Spending Score (1-100)'],
                hue=scaled_df['Cluster_BalancedKMeans'], palette='viridis', ax=ax[1], legend=False)
ax[1].set_title("Balanced K-Means Clustering")

plt.show()


