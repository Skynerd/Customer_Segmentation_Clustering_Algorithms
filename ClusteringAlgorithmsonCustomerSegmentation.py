import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Load dataset
file_csv = 'Mall_Customers.csv'
data = pd.read_csv(file_csv)

# Preview the dataset
data.head()


# Checking for missing values
print(data.isnull().sum())

# Statistical summary
print(data.describe())

# Visualize distributions
dataTitle = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
dataTotal = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
sns.pairplot(dataTotal)
plt.show()

wcsslist = [[]]*len(dataTitle)*len(dataTitle)
dwcsslist = [[]]*len(dataTitle)*len(dataTitle)
ddwcsslist = [[]]*len(dataTitle)*len(dataTitle)
nClusterlist = [20]*len(dataTitle)*len(dataTitle)
X_scaled = [[]]*len(dataTitle)*len(dataTitle)
titlelist = [[]]*len(dataTitle)*len(dataTitle)

cnt = 0
for dataTitleX in dataTitle:
    for dataTitleY in dataTitle:  

        # Select features for clustering
        titlelist[cnt] = [dataTitleX, dataTitleY]
        X = data[titlelist[cnt]]

        # Standardize the data (Optional)
        scaler = StandardScaler()
        X_scaled[cnt] = scaler.fit_transform(X)


        # Elbow method to find optimal k
        wcss = []
        dwcss = []
        ddwcss = []
        for i in range(1, 20):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X_scaled[cnt])
            wcss.append(kmeans.inertia_)
            if i>1: dwcss.append(wcss[i-1]-wcss[i-2])
            if i>2: 
                ddwcss.append(dwcss[i-2]-dwcss[i-3])
                if ddwcss[i-3] <= 0 and nClusterlist[cnt] == 20 : nClusterlist[cnt] = i - 2
        wcsslist[cnt] = wcss
        dwcsslist[cnt] = dwcss
        ddwcsslist[cnt] = ddwcss
        cnt += 1

# Plot the elbow curve
fig, axes = plt.subplots(len(dataTitle), len(dataTitle), figsize=(10, 10))  
# Plot on each subplot
for i, ax in enumerate(axes.flat):
    ax.plot(range(1, 20), wcsslist[i])  
    ax.plot(nClusterlist[i], wcsslist[i][nClusterlist[i]-1], 'r*')  
    ax.set_title(f'Plot {i + 1}')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('WCSS')
# Display the plot
plt.tight_layout()
plt.show() 

# Plot 1st derivative of WCSS (dwcss/dk)
fig, axes = plt.subplots(len(dataTitle), len(dataTitle), figsize=(10, 10))  
# Plot on each subplot
for i, ax in enumerate(axes.flat):
    ax.plot(range(1, 19), dwcsslist[i])  
    ax.plot(nClusterlist[i], dwcsslist[i][nClusterlist[i]-1], 'r*')  
    ax.set_title(f'Plot {i + 1}')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('d(WCSS)/d(Number of Clusters)')
# Display the plot
plt.tight_layout()
plt.show() 

# Plot 2nd derivative of WCSS (ddwcss/d^2k)
fig, axes = plt.subplots(len(dataTitle), len(dataTitle), figsize=(10, 10))  
# Plot on each subplot
for i, ax in enumerate(axes.flat):
    ax.plot(range(1, 18), ddwcsslist[i])  
    ax.plot(nClusterlist[i], ddwcsslist[i][nClusterlist[i]-1], 'r*')  
    ax.set_title(f'Plot {i + 1}')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('d^2(WCSS)/d(Number of Clusters)^2')
# Display the plot
plt.tight_layout()
plt.show() 
 
# Applying K-means
fig, axes = plt.subplots(len(dataTitle), len(dataTitle), figsize=(10, 10))  
colorlist = ['red','blue','green','cyan','magenta','brown','black','gray'] 
for i, ax in enumerate(axes.flat): 
        kmeans = KMeans(n_clusters=nClusterlist[i], init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(X_scaled[i])

        # Add cluster labels to the original data
        data['Cluster'] = clusters

        # Visualize the clusters
        for j in range(nClusterlist[i]): 
            ax.scatter(X_scaled[i][clusters == j, 0], X_scaled[i][clusters == j, 1], s=10, c=colorlist[j], label=f'Cluster {j}') 

        # Plot centroids
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='yellow', label='Centroids')
        #plt.title('Clusters of Customers')
        ax.set_xlabel(titlelist[i][0])
        ax.set_ylabel(titlelist[i][1])
        ax.legend() 

# Display the plot
plt.tight_layout()
plt.show() 


# Applying DBSCAN for each pair of features
fig, axes = plt.subplots(len(dataTitle), len(dataTitle), figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    if len(X_scaled[i]) > 0:
        dbscan = DBSCAN(eps=0.3, min_samples=5)  # Adjust epsilon and min_samples for better results
        clusters = dbscan.fit_predict(X_scaled[i])

        # Plot DBSCAN clusters
        for j in np.unique(clusters):
            if j != -1:
                ax.scatter(X_scaled[i][clusters == j, 0], X_scaled[i][clusters == j, 1], s=20, label=f'Cluster {j}')
            else:
                ax.scatter(X_scaled[i][clusters == j, 0], X_scaled[i][clusters == j, 1], s=20, c='gray', label='Noise')

        ax.set_xlabel(titlelist[i][0])
        ax.set_ylabel(titlelist[i][1])
        ax.legend(loc='best')
plt.tight_layout()
plt.show()

 
