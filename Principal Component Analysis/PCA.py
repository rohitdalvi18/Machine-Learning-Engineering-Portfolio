import pandas as pd                         # data manipulation
import numpy as np                          # numerical operations
import matplotlib.pyplot as plt             # basic plotting
import seaborn as sns                       # statistical vizualisation
from sklearn.datasets import load_wine      # load the wine dataset from Scikit-Learn

# Load the dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# dimensions of the dataframe
print(df.shape)

# display the top 5 rows
df.head(10)

# Plot histograms of original features
df.hist(figsize=(14, 10), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions Before Standardization")
plt.show()

# the sklearn approach is using variance with ddof=0 internally,
# which results in slightly different standard deviations compared to Pandas.
# This leads to small differences when computing the covariance matrix.
# meaning N versus N-1

# Standardize the data
# df.std(ddof=1) computes the sample standard deviation (divides by N-1) to
# correct for bias when estimating population variance from a sample.
# X_scaled = (df - df.mean()) / df.std(ddof=1)

# # Convert back to DataFrame for easy visualization
# df_scaled = pd.DataFrame(X_scaled, columns=wine.feature_names)

from sklearn.preprocessing import StandardScaler

# Initialize and apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Convert back to DataFrame for easy visualization
df_scaled = pd.DataFrame(X_scaled, columns=wine.feature_names)

# Check mean and standard deviation
print("Mean after standardization:\n", df_scaled.mean().round(2))
print("\nStandard deviation after standardization:\n", df_scaled.std().round(2))

# Plot histograms after standardization
df_scaled.hist(figsize=(14, 10), bins=20, edgecolor='black', color='green')
plt.suptitle("Feature Distributions After Standardization")
plt.show()

# Compute covariance matrix
# Transpose because the cov() function requires
cov_matrix = np.cov(X_scaled.T)  # Transpose scaled matrix to get correct shape

# Convert to DataFrame for visualization
cov_df = pd.DataFrame(cov_matrix, index=wine.feature_names, columns=wine.feature_names)

# Plot heatmap of covariance matrix
plt.figure(figsize=(10,6))
sns.heatmap(cov_df, cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Covariance Matrix of Wine Dataset")
plt.show()

# Compute eigenvalues & eigenvectors from the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Convert to DataFrame for easier interpretation
# Remember: eigenvalues will be equivalent to the number of columns in your original
# dataset and tell you the amount of variance captured by each PC
# So, we rank these and determine the "cut point" for the components
eigen_df = pd.DataFrame({'Eigenvalue': eigenvalues})
eigen_df = eigen_df.sort_values(by="Eigenvalue", ascending=False)  # Sort descending
eigen_df

# Compute explained variance ratio (proportion of variance captured by each PC)
explained_variance_ratio = eigenvalues / sum(eigenvalues)

# Compute cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot scree plot to visualize explained variance
plt.figure(figsize=(8,5))
plt.plot(range(1, 14), cumulative_variance, marker='o')

# Labels and formatting
plt.xlabel("Number of Principal Components")  # X-axis: Number of components
plt.ylabel("Cumulative Explained Variance")  # Y-axis: Total variance captured so far
plt.title("Scree Plot: Explained Variance vs. Components")  # Title of plot
plt.grid(True)  # Add grid for readability

# Show the plot
plt.show()

# Sort eigenvalues & eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

# Select first 2 components
top_eigenvectors = eigenvectors[:, sorted_indices[:2]]

# Project data onto first two principal components
X_pca_manual = np.dot(X_scaled, top_eigenvectors)

# Convert to DataFrame
df_pca_manual = pd.DataFrame(X_pca_manual, columns=['PC1', 'PC2'])
df_pca_manual.head(200)

# Visualizing the projection

plt.figure(figsize=(8,6))
sns.scatterplot(x=df_pca_manual['PC1'], y=df_pca_manual['PC2'], hue=wine.target, palette='viridis', edgecolor='black')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Wine Dataset (Manual Calculation)')
plt.legend(title='Wine Class')
plt.grid(True)
plt.show()

from sklearn.decomposition import PCA

# Apply PCA using sklearn
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_scaled)

# Convert to DataFrame
df_pca_sklearn = pd.DataFrame(X_pca_sklearn, columns=['PC1', 'PC2'])

# Compare first few rows
df_pca_sklearn.head()

# Compare manual and sklearn results
plt.figure(figsize=(10,5))

# Manual PCA
plt.subplot(1,2,1)
sns.scatterplot(x=df_pca_manual['PC1'], y=df_pca_manual['PC2'], hue=wine.target, palette='viridis', edgecolor='black')
plt.title("Manual PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")

# Sklearn PCA
plt.subplot(1,2,2)
sns.scatterplot(x=df_pca_sklearn['PC1'], y=df_pca_sklearn['PC2'], hue=wine.target, palette='viridis', edgecolor='black')
plt.title("Sklearn PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()

# Sort eigenvalues & eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]

# Select first 10 components
top_eigenvectors = eigenvectors[:, sorted_indices[:10]]

# Project data onto first ten principal components
X_pca_manual = np.dot(X_scaled, top_eigenvectors)


# Convert to DataFrame
df_pca_manual = pd.DataFrame(X_pca_manual, columns=['PC1', 'PC2', 'PC3', 'PC4',
                                                    'PC5', 'PC6', 'PC7', 'PC8',
                                                    'PC9', 'PC10'])
df_pca_manual.head(10)

from sklearn.decomposition import PCA

# Compute PCA on all components first
pca_full = PCA(n_components=None) # Automatically keeps all components
X_pca_full = pca_full.fit_transform(X_scaled)

# Find optimal number of components using explained variance
explained_var_ratio = np.cumsum(pca_full.explained_variance_ratio_) # Cumulative variance

# Plot scree plot
plt.figure(figsize=(8,5))
plt.plot(range(1, 14), explained_var_ratio, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Scree Plot: Explained Variance vs. Components")
plt.grid(True)
plt.show()

# Get optimal components directly for comparison
optimal_components = np.argmax(explained_var_ratio >= 0.95) + 1  # Find where variance >= 95%

print(f"Optimal number of components for 95% variance: {optimal_components}")

# Apply PCA with optimal components
pca_opt = PCA(n_components=optimal_components)
X_pca_opt = pca_opt.fit_transform(X_scaled)

print(f"Explained variance retained with {optimal_components} components:", sum(pca_opt.explained_variance_ratio_))

# Create a df similar to the above where we see projected data for each PC
# Create column names for the DataFrame
column_names = [f'PC{i+1}' for i in range(optimal_components)]

# Create the DataFrame
df_pca_opt = pd.DataFrame(X_pca_opt, columns=column_names)

# Display the first 10 rows (or any number you prefer)
df_pca_opt.head(10)

# Extract loadings (how features contribute to principal components)


# Convert to DataFrame
loadings_df = pd.DataFrame(loadings, index=wine.feature_names, columns=[f'PC{i+1}' for i in range(optimal_components)])

# Display the first few rows of loadings to examine
# These are standardized using the eigenvalues and eigenvectors
# The loadings are the original dataset interpreted in terms of the new features
# (i.e., linear combinations of the original features) you have created

# Result of multiplying the eigenvector with the square root of the eigenvalue to
# bring back the scale of the variance

# Remember: our data was standardized going in and were therefore comparable across
# variables

# We then squished everything down to unit vectors and need to apply back the
# eigenvalue to apply variance of each of the variables
loadings_df

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Plot a circle for reference
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
ax.add_artist(circle)

# Loop over each feature in the dataset to plot its contribution to the principal components
for i, feature in enumerate(wine.feature_names):
    # Draw an arrow from the origin (0,0) to the loading coordinates (PC1, PC2)
    # Loadings represent how much each original feature contributes to the new principal component space
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],  # Arrow starts at (0,0) and ends at (PC1, PC2) coordinates
              color='red',  # Make arrows red for visibility
              alpha=0.7,     # Set transparency to make arrows slightly faded
              head_width=0.02)  # Define the width of the arrowhead for visibility

    # Label each arrow with the corresponding feature name to indicate its contribution
    plt.text(loadings[i, 0] * 1.1,  # Position text slightly away from arrow tip in the PC1 direction
             loadings[i, 1] * 1.1,  # Position text slightly away from arrow tip in the PC2 direction
             feature,               # Use the feature name from wine.feature_names
             color='black',         # Black text for better contrast
             fontsize=12)           # Adjust font size for readability

# Formatting
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Variable Loading Plot (R-Style)")
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)

plt.show()

#Load the data

from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=True)

# Normalize pixel values to [0, 1]
# Why is normalizing ok rather than standardizing?
X = mnist.data.astype(np.float32) / 255.0

# Create a DataFrame (optional) for initial data inspection
df_images = pd.DataFrame(X)

# Extract data into a NumPy array for PCA
X_array = df_images.values

print(df_images.shape)
df_images.head()

# Select 1000 images for PCA transformation
num_samples = 1000
X_subset = X_array[:num_samples]

# Define different numbers of components to test
component_list = [10, 50, 150, 300]
num_display = 10  # Show only 10 images

# Create subplots for original and reconstructed images
fig, axes = plt.subplots(len(component_list) + 1, num_display, figsize=(12, 8))

# Plot original images (first row)
for i in range(num_display):
    axes[0, i].imshow(X_subset[i].reshape(28, 28), cmap='gray')
    # X_subset[i] is a flattened image (a 1D array of 784 pixels, because 28 * 28 = 784).
	  # .reshape(28, 28) converts this 1D array back into a 2D array so imshow() can display it as a 28x28 pixel image.
	  # cmap='gray' shows the image in grayscale, where 0 represents black and higher values represent lighter shades.
    axes[0, i].axis('off') # Hide axis labels
axes[0, 0].set_title("Original Images")

# Perform PCA for different component numbers and reconstruct images
for row, n_components in enumerate(component_list):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_subset)
    X_reconstructed = pca.inverse_transform(X_pca)

    for i in range(num_display):
        axes[row + 1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
        # same as above except using X_reconstructed from PCA instead of original image
        axes[row + 1, i].axis('off') # Hide axis labels

    axes[row + 1, 0].set_title(f"{n_components} Components")

# Show plot
plt.suptitle("PCA Reconstruction Quality at Different Component Levels")
plt.show()

# Assuming mnist.target and df_images are already defined
y = mnist.target.astype(int)  # Convert labels to integers
df_images['label'] = y

# Select specific digits for comparison (e.g., "1" and "3")
digits_to_compare = [1, 3]

# Create subplots to visualize explained variance for each digit
fig, axes = plt.subplots(1, len(digits_to_compare), figsize=(12, 5))

# Initialize variables to find global y-axis limits
min_variance = 1.0
max_variance = 0.0

# First loop to calculate explained variance and find global limits
explained_variances = []  # Store variances to reuse in the next loop

for digit in digits_to_compare:
    # Extract subset of data for the given digit and drop values (i.e., digit label) column
    X_digit = df_images[df_images['label'] == digit].drop(columns=['label']).values

    # Apply PCA
    pca = PCA().fit(X_digit)

    # Compute cumulative explained variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    explained_variances.append(explained_variance)

    # Update global min and max variance for y-axis scaling
    min_variance = min(min_variance, explained_variance.min())
    max_variance = max(max_variance, explained_variance.max())

# Second loop to plot with consistent y-axis
for i, (digit, explained_variance) in enumerate(zip(digits_to_compare, explained_variances)):
    axes[i].plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    axes[i].axhline(y=0.95, color='r', linestyle='--', label="95% variance threshold")
    axes[i].set_xlabel("Number of Principal Components")
    axes[i].set_ylabel("Cumulative Explained Variance")
    axes[i].set_title(f"Explained Variance for Digit {digit}")
    axes[i].legend()
    axes[i].grid()

    # Apply consistent y-axis limits
    axes[i].set_ylim(min_variance, max_variance)

plt.suptitle("PCA Explained Variance Comparison for Different Digits")
plt.show()

!pip install scanpy

import scanpy as sc

# Load RNA-seq data (Paul et al., 2015 dataset)
adata = sc.datasets.paul15()
df = pd.DataFrame(adata.X, columns=adata.var_names)
print("Dataset shape:", df.shape)
print("\nNumber of genes:", len(df.columns))
df.head()

# Perform PCA
# scale the data with StandardScaler()


# Compute PCA on all components first


# Find optimal number of components using explained variance
# Cumulative variance

# Plot scree plot


# Get optimal components directly for comparison
# Find where variance >= 95%

print(f"Optimal number of components for 95% variance: {optimal_components}")

# Apply PCA with optimal components


print(f"Explained variance retained with {optimal_components} components:", sum(pca_opt.explained_variance_ratio_))

# Create a df similar to the above where we see projected data for each PC
# Create column names for the DataFrame
column_names = [f'PC{i+1}' for i in range(optimal_components)]

# Create the DataFrame
df_pca_opt = pd.DataFrame(X_pca_opt, columns=column_names)

# Display the first 10 rows (or any number you prefer)
df_pca_opt.head(10)

# this is the df transposed where each nested list is a single variable
toy_wine = [[12, 13, 12, 14, 14],[5, 2, 3, 2, 3],[0, 1, 1, 2, 1]]

# Average each sublist (get the sample mean of each variable)
sum_vars = np.matrix([[65], [15], [5]])
avg_vars = 1/5 * sum_vars
print(avg_vars)

# Convert toy_wine to a NumPy array for easier manipulation
toy_wine_np = np.array(toy_wine)

# Subtract avg_vars from each column
# This is like standardizing the data (centering the mean)
centered_data = toy_wine_np - avg_vars

# Result back as a list of lists:
A = centered_data.tolist()
A = np.matrix(A)

# dot product (multiply row by column entries and sum)
# row_number by column_number final entry value in row_number, column_number position
# 2nd row by 3rd column final value in (2, 3) position in final matrix
C = A*(A.transpose())

C = 1/5*C
# Covariance matrix
print(C)

# Need to find the eigenvectors (the component loadings i.e., the contribution of
# each variable to a particular component)
eigenvalues, eigenvectors = np.linalg.eig(C)
print(eigenvalues)
print(eigenvectors)

# Compute the loadings (eigenvectors scaled by sqrt of eigenvalues)
loadings = eigenvectors * np.sqrt(np.diag(eigenvalues))

# Project the centered data onto the principal components
projected_data = eigenvectors.T * A

# Print results
print("Loadings:\n", loadings)
print("\nProjected Data:\n", projected_data)


