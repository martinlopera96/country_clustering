import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

path = r'C:\Users\marti\Desktop\MARTÍN\DATA SCIENCE\Platzi\ML_projects\clustering\project\Country-data.csv'
df = pd.read_csv(path)

df.describe()
df.head(5)
df.info()
print(df.shape)
print(df.dtypes)
df.isnull().sum()

# Distribution analysis
int_cols = df.select_dtypes(exclude='object').columns

for i in int_cols:
    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df, y=i)
    plt.show()

fig = plt.figure(figsize=(15, 10))
sns.heatmap(df.drop('country', axis=1).corr(), annot=True, cmap='coolwarm')
plt.show()

# Data Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop('country', axis=1))

df_scaled = pd.DataFrame(df_scaled, columns=df.drop('country', axis=1).columns)

df_scaled.head(5)

# PCA
pca = PCA()
pca.fit(df.drop('country', axis=1))
pca_scaled = pca.transform(df_scaled)

print(pca_scaled)

var = pca.explained_variance_ratio_
print(var)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(var) + 1), var, marker='o', linestyle='-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

cum_var = np.cumsum(np.round(var, decimals=4) * 100)
plt.figure(figsize=(8, 8))
plt.plot(cum_var, 'r-x')
plt.show()

pca_standard = pd.DataFrame(pca_scaled)
pca_standard.drop([4, 5, 6, 7, 8], axis=1, inplace=True)
pca_standard.head(5)

# Clustering

# K-means with PCA
sum_of_squared_distances = []
silhouette_scores = []
K = range(2, 15)
for _ in K:
    km = KMeans(n_clusters=_)
    y = km.fit_predict(pca_standard)
    sum_of_squared_distances.append(km.inertia_)
    silhouette_scores.append(silhouette_score(pca_standard, y))

plt.figure(figsize=(6, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(silhouette_scores, 'rx-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.show()

km = KMeans(n_clusters=4)
y = km.fit_predict(pca_standard)
print(silhouette_score(pca_standard, y))
df['k_means_pca'] = y

# K-means without PCA
sum_of_squared_distances = []
silhouette_scores = []
K = range(2, 15)
for _ in K:
    km = KMeans(n_clusters=_)
    y = km.predict(df_scaled)
    sum_of_squared_distances.append(km.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, y))

plt.figure(figsize=(6, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.show()

km = KMeans(n_clusters=5)
y = km.fit_predict(df_scaled)
print(silhouette_score(df_scaled, y))
df['k_means'] = y

# Agglomerate hierarchical clustering with PCA
plt.figure(figsize=(8, 8))
dendrogram_plot_pca = dendrogram(linkage(pca_standard.values, method='ward'))
plt.title('Country Dendrogram')
plt.xlabel('Clusters')
plt.ylabel('Euclidean Distance')
plt.show()

hc_pca = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
y_hc_pca = hc_pca.fit_predict(pca_standard)
df['hc_pca'] = y_hc_pca

# Agglomerate hierarchical clustering without PCA
plt.figure(figsize=(8, 8))
dendrogram_plot = dendrogram(linkage(df_scaled.values, method='ward'))
plt.title('Country Dendrogram')
plt.xlabel('Clusters')
plt.ylabel('Euclidean Distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(df_scaled)
print(silhouette_score(df_scaled, y_hc))
df['hc'] = y_hc

# DBSCAN with PCA
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(pca_standard)
distances, indices = neighbors_fit.kneighbors(pca_standard)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]
plt.figure(figsize=(6, 6))
plt.plot(distances)
plt.show()

eps_values = np.arange(0.5, 2.0, 0.10)
min_samples = np.arange(3, 12)

dbscan_params = list(product(eps_values, min_samples))
n_of_clusters = []
sil_score = []
for p in dbscan_params:
    y_dbscan = DBSCAN(eps=p[0], min_samples=p[1]).fit_predict(pca_standard)
    try:
        sil_score.append(silhouette_score(pca_standard, y_dbscan))
    except:
        sil_score.append(0)
    n_of_clusters.append(len(np.unique(y_dbscan)))

df_param_tuning = pd.DataFrame.from_records(dbscan_params, columns=['Eps', 'Min_samples'])
df_param_tuning['sil_score'] = sil_score
df_param_tuning['n_clusters'] = n_of_clusters

pivot_1 = pd.pivot_table(df_param_tuning, values='sil_score', columns='Eps', index='Min_samples')
pivot_2 = pd.pivot_table(df_param_tuning, values='n_clusters', columns='Eps', index='Min_samples')

fig_1, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot_1, annot=True, annot_kws={'size': 10}, cmap='coolwarm', ax=ax)
plt.show()

fig_2, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot_2, annot=True, annot_kws={'size': 10}, cmap='coolwarm', ax=ax)
plt.show()

dbscan_train = DBSCAN(eps=1.2, min_samples=3)
y_dbscan = dbscan_train.fit_predict(pca_standard)
print(silhouette_score(pca_standard, y_dbscan))
df['dbscan_pca'] = y_dbscan

# DBSCAN without PCA
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(df_scaled)
distances, indices = neighbors_fit.kneighbors(df_scaled)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]

plt.figure(figsize=(6, 6))
plt.plot(distances)
plt.show()

eps_values = np.arange(1, 3.0, 0.1)
min_samples = np.arange(3, 12)

dbscan_params = list(product(eps_values, min_samples))
n_of_clusters = []
sil_score = []

for p in dbscan_params:
    y_dbscan = DBSCAN(eps=p[0], min_samples=p[1]).fit_predict(df_scaled)
    try:
        sil_score.append(silhouette_score(df_scaled, y_dbscan))
    except:
        sil_score.append(0)
    n_of_clusters.append(len(np.unique(y_dbscan)))

df_param_tuning = pd.DataFrame.from_records(dbscan_params, columns=['Eps', 'Min_samples'])
df_param_tuning['sil_score'] = sil_score
df_param_tuning['n_clusters'] = n_of_clusters

pivot_1 = pd.pivot_table(df_param_tuning, values='sil_score', columns='Eps', index='Min_samples')
pivot_2 = pd.pivot_table(df_param_tuning, values='n_clusters', columns='Eps', index='Min_samples')

fig_3, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot_1, annot=True, annot_kws={'size': 10}, cmap='coolwarm', ax=ax)
plt.show()

fig_4, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(pivot_2, annot=True, annot_kws={'size': 10}, cmap='coolwarm', ax=ax)
plt.show()

dbscan_train = DBSCAN(eps=1.7, min_samples=3)
y_dbscan = dbscan_train.fit_predict(df_scaled)
print(silhouette_score(df_scaled, y_dbscan))
df['dbscan'] = y_dbscan

# Results evaluation
results = df[
    ['child_mort',
     'exports',
     'health',
     'imports',
     'income',
     'inflation',
     'life_expec',
     'total_fer',
     'gdpp',
     'k_means_pca']
]

sns.pairplot(data=results[
    ['child_mort',
     'exports',
     'health',
     'imports',
     'income',
     'inflation',
     'life_expec',
     'total_fer',
     'gdpp',
     'k_means_pca']],
             hue='kmeans_pca',
             palette='coolwarm'
             )

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x='child_mort', y='gdpp', hue='kmeans_pca', palette='coolwarm')
plt.show()
