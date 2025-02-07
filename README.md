# K-means Clustering Implementation

This project implements K-means clustering algorithm from scratch using Python. The implementation includes clustering for k=2 and k=3, with proper data normalization and visualization.

## Project Description

The K-means clustering algorithm is implemented without using any machine learning libraries (like scikit-learn). The project includes:
- Data normalization
- K-means clustering implementation for k=2 and k=3
- Visualization of clustering results
- Generation of PDF report with results

## Requirements

### Libraries Used
- NumPy: For numerical computations and data manipulation
- Matplotlib: For data visualization
- OS: For file handling

### Installation




## Dataset
The program expects a dataset file named 'dataset.txt' in the same directory as the script. The dataset should be in the following format:
- Two-dimensional data points
- Space or comma-separated values
- Each line represents one data point

## Implementation Details

### Features
1. Data Normalization using min-max scaling
2. Custom implementation of K-means clustering
3. Euclidean distance calculation
4. Centroid initialization and updates
5. Cluster assignment
6. Visualization with proper labeling

### Functions
- `euclidean_distance(x1, x2)`: Calculates distance between two points
- `assign_clusters(data, centroids)`: Assigns data points to nearest centroid
- `update_centroids(data, clusters, k)`: Updates centroid positions
- `kmeans_clustering(data, k, max_iterations)`: Main clustering function

## Output
The program generates:
1. Two plots:
   - K-means clustering with k=2
   - K-means clustering with k=3
2. PDF report containing both visualizations
3. Printed cluster centers (denormalized) for both k values


## Results
The output includes:
- Normalized feature plots
- Cluster assignments
- Centroid positions
- PDF report with visualizations


## Author
[Swastideep Khuntia]

## License
This project is licensed under the MIT License - see the LICENSE.md file for details
