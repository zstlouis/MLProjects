import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# load data into dataframe
df = pd.read_csv('Mall_Customers.csv')

print(df.head())

# finding number of rows and columns
print(df.shape)

# check if any data is missing
print(df.isnull().sum())

# get values for last 2 columsn (Annual Income, Spending Score)
X = df.iloc[:, 3:].values
print(X)

# within cluster sum of squares
# want to find min values based on different cluster sizes
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    # gives wcss value for each cluster size
    wcss.append(kmeans.inertia_)

# plot elbow point graph
sns.set()
plt.plot(range(1,11), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# optimum number of clusters is 5
# training the k-means clustering model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point in their cluster
Y = kmeans.fit_predict(X)
print(Y)

# plot the points
plt.figure(figsize=(8,8))
# X[y==0,0] -> avg income for the customers that belong to group 1 (y==0) from kmeans.fit_predict
# X[y==0,1] -> spending score for the customers that belong to group 1(y==0) from kmeans.fit_predict
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='red', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='pink', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='yellow', label='Cluster 5')

# plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=75, c='black', label='centroid')

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title('Customer Clustering')
plt.show()
