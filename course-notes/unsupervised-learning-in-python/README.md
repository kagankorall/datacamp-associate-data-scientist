# Unsupervised Learning in Python

## Course Overview
This course covers unsupervised learning techniques including clustering algorithms, dimensionality reduction, and pattern discovery in unlabeled data.

## Key Topics Covered

### 1. Clustering Algorithms
- k-Means clustering
- Hierarchical clustering
- DBSCAN
- Gaussian Mixture Models

### 2. Dimensionality Reduction
- Principal Component Analysis (PCA)
- t-SNE
- Linear Discriminant Analysis (LDA)
- Feature selection techniques

### 3. Association Rules
- Market basket analysis
- Apriori algorithm
- Support, confidence, and lift

### 4. Anomaly Detection
- Isolation Forest
- One-Class SVM
- Statistical methods

## Key Concepts

### Clustering Example
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate clustering
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
dbscan_score = silhouette_score(X_scaled, dbscan_labels)

print(f"k-Means Silhouette Score: {kmeans_score:.3f}")
print(f"DBSCAN Silhouette Score: {dbscan_score:.3f}")
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
ax1.set_title('PCA')

ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels)
ax2.set_title('t-SNE')

plt.show()
```

### Optimal Number of Clusters
```python
from sklearn.metrics import silhouette_score

# Elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(k_range, inertias, 'bo-')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')

ax2.plot(k_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Number of clusters')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')

plt.show()
```

### Anomaly Detection
```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)

# One-Class SVM
one_class_svm = OneClassSVM(nu=0.1)
svm_labels = one_class_svm.fit_predict(X_scaled)

# Count anomalies
print(f"Isolation Forest anomalies: {(anomaly_labels == -1).sum()}")
print(f"One-Class SVM anomalies: {(svm_labels == -1).sum()}")
```

### Association Rules (Simple Example)
```python
# Market basket analysis example
from itertools import combinations

def calculate_support(itemset, transactions):
    """Calculate support for an itemset."""
    count = sum(1 for transaction in transactions if itemset.issubset(set(transaction)))
    return count / len(transactions)

def find_frequent_itemsets(transactions, min_support=0.1):
    """Find frequent itemsets using simple approach."""
    # Get all unique items
    all_items = set()
    for transaction in transactions:
        all_items.update(transaction)
    
    # Find frequent 1-itemsets
    frequent_itemsets = []
    for item in all_items:
        support = calculate_support({item}, transactions)
        if support >= min_support:
            frequent_itemsets.append(({item}, support))
    
    return frequent_itemsets

# Example transactions
transactions = [
    ['bread', 'milk', 'eggs'],
    ['bread', 'butter'],
    ['milk', 'butter', 'cheese'],
    ['bread', 'milk', 'butter'],
    ['eggs', 'cheese']
]

frequent_items = find_frequent_itemsets(transactions, min_support=0.2)
print("Frequent itemsets:")
for itemset, support in frequent_items:
    print(f"{itemset}: {support:.2f}")
```

## Course Notes

# Clustering for Dataset Exploration

## Unsupervised Learning

Unsupervised learning finds patterns in data. For example, clustering customers by their purchases. Compressing the data using purchase patterns (dimension reduction). In supervised learning, user finds patterns for a prediction task. On the other hand, in unsupervised learning user finds patterns in a data without specific prediction task in mind.

For example, iris data is 4 dimensional and dimension means number of features. Dimension is too high to visualize. 

### k-means clustering

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)

labels = model.predict(samples)

import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()
```

Without starting over, new samples can be assigned to existing clusters. k-means does this with remembering the mean of each cluster (the centroids)

```python
# Example
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()
```

## Evaluating a Clustering

Clusters vs species is a cross-tabulation.

```python
import pandas as pd
df = pd.DataFrame({'labels':labels, 'species':species})

ct = pd.crosstab(df['labels'],df['species'])
```

### Inertia

Inertia measures clustering quality. It measures how spread out the clusters are which means lower inertia is better quality. It is distance from each sample to centroid of its cluster. k-means attempts to minimize the inertia when choosing clusters.

```python
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)
model.fit(samples)
print(model.inertia_)
```

A good clustering has tight clusters which means low inertia but also not too many clusters. Therefore, we need a trade-off. 

```python
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```

## Transforming Features for Better Clusterings

We transform features because sometimes features can have very different variances. Variance of a feature measures spread of its values. 

In k-means: feature variance = feature influence

StandardScaler transforms each feature to have mean 0 and variance 1. With this way features are said to be standardized. 

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std = True)
samples_scaled = scaler.transform(samples)
```

StandardScaler and KMeans have similar methods. StandardScaler use fit then transform. On the other hand, KMeans uses fit then predict methods.

```python
# Using pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)

labels = pipeline.predict(samples)
```

```python
# Example
# Import Normalizer
from sklearn.preprocessing import Normalizer
# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
```

# Visualization with Hierarchical Clustering and t-SNE

## Visualizing Hierarchies

t-SNE: Creates a 2D map of a dataset.

### Hierarchical Clustering

Every country begins in a separate cluster. At each step, the tow closest clusters are merged. Continue until all countries in a single cluster. This is agglomerative hierarchical clustering. Dendrogram shows the visualization of this clustering. In dendrogram, read from the bottom up. Vertical lines represent clusters. 

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
mergings = linkage(samples, method = 'complete')
dendrogram = mergings,
						 labels = country_names,
						 leaf_rotation = 90,
						 leaf_font_size = 6)
plt.show()
```

## Cluster Labels in Hierarchical Clustering

Height on dendrogram = distance between merging clusters. Height on dendrogram specifies maximum distance between merging clusters.

Distance between clusters defined by a linkage method. In “complete” linkage: distance between clusters is maximum distance between their samples. This specified via method parameter.

```python
from scipy.cluster.hierarchy import linkage
mergings = linkage(samples, method = 'complete')
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion = 'distance')

import pandas as pd
pairs = pd.DataFrame({'labels':labels, 'countries':country_names})
print(pairs.sort_values('labels'))
```

## t-SNE for 2D Maps

t-SNE = t-distributed stochastic neighbor embedding 

Maps samples to 2D spaces (or 3D)

Map approximately preserves nearness of samples. 

Great for inspecting datasets.

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()
```

t-SNE has only fit_transform method. It simultaneously fits the model and transforms the data.

# Decorrelating Your Data and Dimension Reduction

## Visualizing the PCA Transformation

Dimension reduction is more efficient in storage and computation. It also removes less-informative (noisy) features. 

### Principal Component Analysis (PCA)

Fundamental dimension reduction technique. 

It has two steps: first step is decorrelation, second step is reduces dimension.

PCA aligns data with axes. It rotates data samples to be aligned with axes. It shifts data samples so they have mean 0. No information is lost.

PCA is a scikit-learn component like KMeans or StandardScaler. fit learns the transformation from given data. transform applies the learned transformation. In particular, transform can also be applied to new data.

```python
from sklearn.decomposition import PCA
model = PCA()
model.fit(samples)

transformed = model.transform(samples)

print(model_components_)
```

PCA features:

- Rows of transformed correspond to samples
- Columns of transformed are the PCA features

Linear correlation can be meaused with Pearson correlation. It takes values between -1 and 1. Value of 0 means no linaer correlation.

Principal component = directions of variance

![image.png](attachment:0a872a06-f5ba-4d28-9768-cb6dfc550dea:image.png)

```python
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)
```

## Intrinsic Dimension

Intrinsic dimension of a dataset means number of features needed to approximate the dataset. Essential idea behind dimesion reduction. 

PCA identifies intrinsic dimension. Scatter plots work only if samples have 2 or 3 features.

Intrinsic dimension is number of PCA features with significant variance. Variance is a significant factor. Therefore, according to the below graph features 0 and 1 have important variance.

![image.png](attachment:f763be6f-81b0-46b7-ad0d-6db84a969eaa:image.png)

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(samples)

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
```

Intrinsic dimension is an idealization but there is not always one correct answer.

```python
# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()
```

```python
# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
```

## Dimension Reduction with PCA

![image.png](attachment:8cf6fe71-9754-4a74-b603-00172f9c02f7:image.png)

To do dimension reduction with PCA, we have to specify how many features to keep. For example, PCA(n_components=2) which means it keeps the first 2 PCA features.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(samples)

transformed = pca.transform(samples)
print(transformed.shape)
```

PCA discards low variance PCA features. It assumes the high variance features are informative.

# Discovering Interpretable Features

## Non-negative Matrix Factorization (NMF)

Unlike PCA, NMF models are interpretable. Easy to interpret means easy to explain. However, all sample features must be non-negative.

NMF is scikit-learn follows fit() and transform() pattern like PCA, but user must specift number of components.

```python
from sklearn.decomposition import NMF
model = NMF(n_components = 2)
model.fit(samples)

nmf_features = model.transform(samples)
```

NMF has components just like PCA has principal components. Dimension of components = dimesion of samples. Entries are non-negative. In sample reconstruction, multiply components by feature values, and add up. Can also be expressed as a product of matrices.

## NMF Learns Interpretable Parts

```python
from sklearn.decomposition import NMF
model = NMF(n_components = 10)
model.fit(articles)

print(nmf.components_.shape) -> (10,800)
```

For images, NMF components are parts of images.

```python
# Example
# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape((13,8))

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()
```

## Building Recommender System Using NMF