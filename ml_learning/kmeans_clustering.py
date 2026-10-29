
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. Create sample data
X, y_true = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# 2. Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# 3. Train model and predict clusters
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 4. Print cluster information
print("First 10 Cluster Labels:")
print(labels[:10])

print("\nCentroids:")
print(centroids)

print("\nNumber of points in each cluster:")
for i in range(3):
    print(f"Cluster {i}: {sum(labels == i)} points")

# 5. Plot clusters
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    s=50
)

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    color="red",
    label="Centroids"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering (K=3)")
plt.legend()
plt.show()
