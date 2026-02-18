# Intelligent Customer Segmentation Project
# Author: Your Name
# Description: Segment customers using K-Means clustering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
print("Loading dataset...")
df = pd.read_csv('data/Mall_Customers.csv')

print("\nDataset Preview:")
print(df.head())

# -----------------------------
# 2️⃣ Select Features
# -----------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3️⃣ Elbow Method
# -----------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# -----------------------------
# 4️⃣ Apply KMeans (5 clusters)
# -----------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# -----------------------------
# 5️⃣ Evaluate Model
# -----------------------------
score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {score:.2f}")

# -----------------------------
# 6️⃣ Visualize Clusters
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set2'
)

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

print("\nCustomer segmentation completed successfully!")
