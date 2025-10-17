# %%
import numpy as np
from sklearn.datasets import load_wine

# load the wine dataset as a dictionary-like object
wine = load_wine()
print("Wine dataset keys:", wine.keys())
print("Wine dataset description:", wine.DESCR) # Description of the dataset

X = wine.data # numpy.ndarray
y = wine.target # numpy.ndarray

print("Wine dataset feature matrix:", X)
print("Wine dataset target vector:", y)

# %%
def k_means(data_points, centers_init, n_clusters, max_iterations=100, tol=1e-4):
  centers = centers_init.copy() # make a copy of initial center to work on.
  for _ in range(max_iterations): # The underscore _ is a throwaway variable, meaning “I don’t care about the loop variable.”

    # Compute squared Euclidean distances to each centroid
    # Result shape: (n_samples, k)
    distances = np.linalg.norm(data_points[:, np.newaxis] - centers, axis=2)

    # Assign each point to the index of the closest centroid
    closest_centroids = np.argmin(distances, axis=1)

    # Update centroids to be the mean of the data points assigned to them
    new_centers = np.zeros((n_clusters, data_points.shape[1]))
    # End if centroids no longer change
    for i in range(n_clusters):
      new_centers[i] = data_points[closest_centroids == i].mean(axis=0)

    if np.linalg.norm(new_centers - centers) < tol:
      break
    centers = new_centers
  return centers, closest_centroids

# %%
J = {} # Dictionary to store within-cluster variance for each k
np.random.seed(42)  # For reproducibility
for n_clusters in range(2, 11):

    # Randomly initialize cluster centers by selecting k unique data points
    centers_init = X[np.random.choice(X.shape[0], n_clusters, replace=False)]  
    # np.random.choice(): Randomly selects n_clusters unique indices from 0 … n_samples-1.
    #                     replace=False ensures no duplicate picks.
    centers, labels = k_means(X, centers_init, n_clusters=n_clusters)
    within_cluster_var = np.sum((X - centers[labels]) ** 2) # sum of squared distances to the assigned centroid.
    J[n_clusters] = within_cluster_var
    print(f"Converged centers for {n_clusters} clusters:", centers)
    print(f"Within-cluster variance for {n_clusters} clusters:", within_cluster_var)


# %%
# Plotting the within-cluster variance over the number of clusters
import matplotlib.pyplot as plt
plt.plot(J.keys(), J.values(), marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster variance')

# %% [markdown]
# Use sklearn's libraries

# %%
from sklearn.cluster import KMeans

J = {}
np.random.seed(42)  # For reproducibility
for n_clusters in range(2, 11):
    centers_init = X[np.random.choice(X.shape[0], n_clusters, replace = False)] # np.random.choice(): k random indices.
    kmeans = KMeans(n_clusters=n_clusters, init=centers_init, n_init=1)
    #n_init: The number of times the KMeans algorithm will run with different centroid seeds
    #        Setting n_init=1 means it will only run once, using the given centers_init
    kmeans.fit(X)
    within_cluster_var = np.sum((X - kmeans.cluster_centers_[kmeans.labels_]) ** 2)
    J[n_clusters] = within_cluster_var
    print(f"Within-cluster variance for {n_clusters} clusters:", within_cluster_var)

import matplotlib.pyplot as plt
plt.plot(list(J.keys()), list(J.values()), marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster variance')


