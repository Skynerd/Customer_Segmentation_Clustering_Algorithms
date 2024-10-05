# Customer Segmentation using Clustering Algorithms

This project demonstrates how to perform customer segmentation using unsupervised learning algorithms such as **K-Means** and **DBSCAN**. By clustering customer data based on age, annual income, and spending score, we can uncover patterns and group customers with similar behaviors.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Elbow Method for Optimal Clusters](#elbow-method-for-optimal-clusters)
- [K-Means Clustering](#k-means-clustering)
- [DBSCAN Clustering](#dbscan-clustering)
- [Results](#results)
- [Installation and Usage](#installation-and-usage)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview
Customer segmentation helps businesses tailor marketing strategies to different groups of customers based on their behaviors and demographics. This project clusters customers into distinct groups based on the **Mall Customer Segmentation Dataset**, which includes features like:
- Age
- Annual Income (in $k)
- Spending Score (1-100)

## Dataset
The dataset used in this project is the **Mall Customers Dataset** available from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?resource=download). It contains 200 entries with the following columns:
- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: The customer's Annual income in thousands of dollars.
- **Spending Score (1-100)**: A score based on customer behavior and purchases.

![https://github.com/Skynerd/DigitPredictor/blob/main//DemoPlots/Original Data.png](https://github.com/Skynerd/Customer_Segmentation_Clustering_Algorithms/blob/main/DemoPlots/Original%20Data.png)

## Project Workflow
1. **Data Loading and Preprocessing**: 
   - Load and visualize the dataset.
   - Check for missing values and provide a statistical summary.
   - Standardize the data using `StandardScaler` to prepare for clustering.

2. **Feature Selection**:
   - Clustering is performed on combinations of the following features: `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.

3. **Elbow Method for Optimal Clusters**:
   - Use the **Elbow Method** to determine the optimal number of clusters for K-Means by plotting the within-cluster sum of squares (WCSS).
   
4. **K-Means Clustering**:
   - Apply K-Means clustering to identify customer segments.
   - Visualize the clusters and centroids for different pairs of features.

5. **DBSCAN Clustering**:
   - Apply **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** for discovering clusters of varying shapes,along with noise points.

## Elbow Method for Optimal Clusters
To find the optimal number of clusters for K-Means, we use the **Elbow Method**, where we:
- Calculate the **WCSS (Within-Cluster Sum of Squares)** for different values of `k` (number of clusters).
- Plot WCSS against `k` and identify the "elbow point",which represents the optimal number of clusters.
- The red star indicates the optimal number of clusters for the given plot.
### Elbow Plot
![Elbow Plot](https://github.com/Skynerd/Customer_Segmentation_Clustering_Algorithms/blob/main/DemoPlots/WCSS%20vs%20no.%20of%20Clusters%20(the%20elbow%20curves).png)
### 1st derivative of WCSS
![1st derivative of WCSS](https://github.com/Skynerd/Customer_Segmentation_Clustering_Algorithms/blob/main/DemoPlots/1st%20derivative%20of%20WCSS.png)
### 2nd derivative of WCSS
![2nd derivative of WCSS](https://github.com/Skynerd/Customer_Segmentation_Clustering_Algorithms/blob/main/DemoPlots/2nd%20derivative%20of%20WCSS.png)

## K-Means Clustering
After determining the optimal number of clusters, we apply K-Means clustering on different pairs of features:
- **Age vs Annual Income**
- **Age vs Spending Score**
- **Annual Income vs Spending Score**

The resulting clusters are visualized with different colours, and the centroids are highlighted.

![K-Means Clustering Visualization](https://github.com/Skynerd/Customer_Segmentation_Clustering_Algorithms/blob/main/DemoPlots/K-Means%20Clustering%20Visualization.png)

## DBSCAN Clustering
For more complex datasets, DBSCAN is a great alternative to K-Means, as it is able to:
- Find clusters of arbitrary shapes.
- Identify noise points that don't belong to any cluster.

We apply DBSCAN to the same feature pairs and visualize the results.

![DBSCAN Clustering Visualization](https://github.com/Skynerd/Customer_Segmentation_Clustering_Algorithms/blob/main/DemoPlots/DBSCAN%20Clustering%20Visualization.png)

## Results
The project successfully segments customers into different groups based on behaviour and demographics. This allows targeted marketing strategies such as personalized offers or promotions for specific groups.

Here are sample visualizations:
- **K-Means Clustering**: Customers are segmented based on the selected feature pairs.
- **DBSCAN Clustering**: Shows clusters and noise points, which can identify outliers or special customer cases.

## Installation and Usage

### 1. Clone the Repository
```
git clone https://github.com/yourusername/customer-segmentation-clustering.git
cd customer-segmentation-clustering
```

### 2. Install Dependencies
Make sure you have Python 3.11 installed. Then, install the required Python libraries by running:
```
pip install -r requirements.txt
```
### 3. Run the Jupyter Notebook
To run the project, open the Jupyter Notebook and run the cells in the following order:
```
jupyter notebook customer_segmentation.ipynb
```

Alternatively, you can run the script:
```
python customer_segmentation.py
```

## Future Improvements
- **Advanced Feature Engineering**: Add more relevant features such as customer transaction history or geolocation data to enhance clustering performance.
- **Hierarchical Clustering**: Implement **Hierarchical Clustering** and compare its results with K-Means and DBSCAN.
- **Hyperparameter Tuning**: Further tune DBSCAN parameters like `eps` and `min_samples` to better identify clusters and noise points.
 
