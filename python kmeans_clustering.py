import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages

# Print current working directory and its contents
print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir())

# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("New working directory:", os.getcwd())
print("Files in the new directory:", os.listdir())

# Load the dataset
try:
    data = np.loadtxt('dataset.txt', encoding='utf-8')
    print("Dataset loaded successfully. Shape:", data.shape)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print(f"Attempted to load file from: {os.path.abspath('dataset.txt')}")
    print("Creating sample data...")
    data = np.random.rand(100, 2)

# Normalize the data
data_normalized = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[clusters == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids

def kmeans_clustering(data, k, max_iterations=100):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iterations):
        old_centroids = centroids.copy()
        clusters = assign_clusters(data, centroids)
        centroids = update_centroids(data, clusters, k)
        
        if np.all(old_centroids == centroids):
            break
            
    return clusters, centroids

# Create PDF
pdf = PdfPages('kmeans_clustering_results.pdf')

# K-means for k=2
clusters_k2, centroids_k2 = kmeans_clustering(data_normalized, k=2)

# Plot for k=2
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_normalized[:, 0], data_normalized[:, 1], c=clusters_k2, cmap='viridis')
plt.scatter(centroids_k2[:, 0], centroids_k2[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering (k=2)')
plt.xlabel('Normalized Feature 1')
plt.ylabel('Normalized Feature 2')
plt.colorbar(scatter, label='Cluster')
plt.legend(['Data Points', 'Centroids'])
pdf.savefig()
plt.show()

# Print cluster centers for k=2
print("\nCluster Centers for k=2 (denormalized):")
denorm_centroids_k2 = centroids_k2 * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data, axis=0)
for i, center in enumerate(denorm_centroids_k2):
    print(f"Cluster {i+1}: {center}")

# K-means for k=3
clusters_k3, centroids_k3 = kmeans_clustering(data_normalized, k=3)

# Plot for k=3
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_normalized[:, 0], data_normalized[:, 1], c=clusters_k3, cmap='viridis')
plt.scatter(centroids_k3[:, 0], centroids_k3[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering (k=3)')
plt.xlabel('Normalized Feature 1')
plt.ylabel('Normalized Feature 2')
plt.colorbar(scatter, label='Cluster')
plt.legend(['Data Points', 'Centroids'])
pdf.savefig()
plt.show()

# Print cluster centers for k=3
print("\nCluster Centers for k=3 (denormalized):")
denorm_centroids_k3 = centroids_k3 * (np.max(data, axis=0) - np.min(data, axis=0)) + np.min(data, axis=0)
for i, center in enumerate(denorm_centroids_k3):
    print(f"Cluster {i+1}: {center}")

# Close PDF
pdf.close()
print("\nPDF report generated successfully!")
