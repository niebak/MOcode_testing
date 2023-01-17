import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt

Show_plot=True
USE_KMEANS = False
USE_DBSCAN = True

#Initializing data
train_data, _ = make_classification(n_samples=1000,
                                       n_features=2,
                                       n_informative=2,
                                       n_redundant=0,
                                       n_clusters_per_class=1,
                                       random_state=4)
if USE_KMEANS:
    # Look at the data:
    if Show_plot:
        fig=plt.figure(figsize=[4,3])
        ax0=fig.add_subplot(1,1,1)
        ax0.scatter(train_data[:,0],train_data[:,1])
        ax0.grid()
        ax0.set_title('Looking at the data')
        plt.show()

    # # Found the optimal number of clusters in the dataset
    # X = train_data
    # wcss = []

    # # Fit K-Means to the data for different numbers of clusters and calculate WCSS
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(X)
    #     wcss.append(kmeans.inertia_)

    # # Plot the WCSS values for each number of clusters
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()

    # Initialize the k-means algorithm with 5 clusters
    kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

    # Fit the k-means algorithm to the data
    kmeans.fit(train_data)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Get the coordinates of the cluster centers
    Cluster_centers = kmeans.cluster_centers_
    Clustered_data = kmeans.transform(train_data).argmin(axis=1) # Classify the point into its cluster
    Cluster_labels=np.unique(Clustered_data)

    Show_plot=False # Look at the clusters
    if Show_plot:
        fig=plt.figure(figsize=[4,3])
        ax0=fig.add_subplot(1,1,1)
        for i in Cluster_labels:
            cluster_points = train_data[Clustered_data == i]
            ax0.scatter(cluster_points[:,0],cluster_points[:,1],label=f'Cluster {i}')
            ax0.scatter(Cluster_centers[i,0],Cluster_centers[i,1],label=f'Center for {i}')
        ax0.grid()
        ax0.set_title('Clusters')
        ax0.legend()
        plt.show()
    print(Cluster_labels)
if USE_DBSCAN:
    DBSCAN_model=DBSCAN(eps=0.23,min_samples=9)
    DBSCAN_model.fit(train_data)
    Clustered_data = DBSCAN_model.fit_predict(train_data)
    Cluster_labels = np.unique(Clustered_data)
    if Show_plot:
        fig=plt.figure(figsize=[4,3])
        ax0=fig.add_subplot(1,1,1)
        for cluster in Cluster_labels:
            Cluster_points=train_data[Clustered_data==cluster]
            ax0.scatter(Cluster_points[:,0],Cluster_points[:,1],label=f'Cluster {cluster}')
        ax0.grid()
        ax0.legend()
        plt.show()
# It seems like KMEANS is the best for this dataset, will keep in mind for the database.
#  Still need to solve how to apply this method to the data in order to not mask information