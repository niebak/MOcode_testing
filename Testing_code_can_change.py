import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
import sys
sys.path.append('../DIVCODE')
from code_functions import *
import os

# To limit memory leak, will result in a higher runtime
os.environ["OMP_NUM_THREADS"] = "1"


# defines
Verbose=False
Show_Plot=False
variables = ['position_lat', 'position_long', 'altitude']

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'

TDF=fit_records_to_frame(ownfile,variables)
TDF0=pd.read_excel(coursename)
TDF1=pd.read_parquet(parquetfile)
TDF2=TDF[['position_lat','position_long','altitude']].dropna()
#  Read data
Weird_format=True # If the data is from the xlsx file, it is in a weird format so we need to change this
if Weird_format:
    # The data is in a weird format so we need to change it:
    TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
    TDF0['position_long'] = TDF0['position_long']/(2**32/360)
    TDF2['position_lat'] = TDF2['position_lat']/(2**32/360)
    TDF2['position_long'] = TDF2['position_long']/(2**32/360)

# Create the vector-representation coordinates
TDF0_vector = coordinate_to_vector_dataframe(TDF0)
TDF1_vector = coordinate_to_vector_dataframe(TDF1)
TDF2_vector = coordinate_to_vector_dataframe(TDF2)

TDF0_seg_marker=detect_and_mark_change_in_direction(
    TDF0_vector['Vposition_lat'].tolist(),
    TDF0_vector['Vposition_long'].tolist(),
    TDF0_vector['Valtitude'].tolist())
TDF1_seg_marker=detect_and_mark_change_in_direction(
    TDF1_vector['Vposition_lat'].tolist(),
    TDF1_vector['Vposition_long'].tolist(),
    TDF1_vector['Valtitude'].tolist())
TDF2_seg_marker=detect_and_mark_change_in_direction(
    TDF2_vector['Vposition_lat'].tolist(),
    TDF2_vector['Vposition_long'].tolist(),
    TDF2_vector['Valtitude'].tolist())

TDF0_seg =  marker_to_segment(TDF0_seg_marker, initial_segment=0)
TDF0=pd.concat([TDF0,TDF0_vector],axis=1)
TDF0['segments']=TDF0_seg


TDF1_seg =  marker_to_segment(TDF1_seg_marker, initial_segment=0)
TDF1=pd.concat([TDF1,TDF1_vector],axis=1)
TDF1['segments']=TDF1_seg

TDF2_seg =  marker_to_segment(TDF2_seg_marker, initial_segment=0)
TDF2=pd.concat([TDF2,TDF2_vector],axis=1)
TDF2['segments']=TDF2_seg

# Create a fake database, using segments as trails.
pseudoDB = TDF1[['Vposition_lat','Vposition_long','Valtitude','segments']].groupby('segments').mean()
pseudoDB=pseudoDB.reset_index()
pseudoDB.drop(pseudoDB.index[0],inplace=True)

latest_segment = pseudoDB['segments'].iloc[-1]

SDF0 = TDF0[['Vposition_lat','Vposition_long','Valtitude','segments']].groupby('segments').mean()
SDF0=SDF0.reset_index()
SDF0.drop('segments', axis=1, inplace=True)
SDF0.drop(SDF0.index[0], inplace=True)
print(SDF0[['Vposition_lat','Vposition_long']].max())

SDF1 = TDF1[['Vposition_lat','Vposition_long','Valtitude','segments']].groupby('segments').mean()
SDF1=SDF1.reset_index()
SDF1.drop(SDF1.index[0], inplace=True)
print(SDF1[['Vposition_lat','Vposition_long']].max())


SDF2 = TDF2[['Vposition_lat','Vposition_long','Valtitude','segments']].groupby('segments').mean()
SDF2=SDF2.reset_index()
SDF2.drop('segments', axis=1, inplace=True)
SDF2.drop(SDF2.index[0], inplace=True)

segments=[0]*SDF0.shape[0]
for i in range(SDF0.shape[0]):
    segments[i]=latest_segment
    latest_segment+=1
SDF0['segments']=segments

segments=[0]*SDF2.shape[0]
for i in range(SDF2.shape[0]):
    segments[i]=latest_segment
    latest_segment+=1
SDF2['segments']=segments
temp=pd.concat([SDF0,SDF2],axis=0)
pseudoDB=pd.concat([pseudoDB,temp],axis=0)
print(f'We now have a DB with {pseudoDB.shape[0]} inputs!\n It has these columns: {pseudoDB.columns}')

# Take a peek at the "database"
Show_Plot=True
if Show_Plot:
    fig = plt.figure()
    ax0=fig.add_subplot(1,1,1)
    ax0.scatter(pseudoDB['Vposition_lat'],pseudoDB['Vposition_long'])
    ax0.grid()
    plt.show()



# By using the Elbow method that is shown in the clustering script(see OldTestCode)
# We can see that the optimal number of clusters are 3 or 4
# train_data = pseudoDB[['Vposition_lat','Vposition_long']].values
# DBSCAN_model=DBSCAN(eps=0.0005,min_samples=9)
# DBSCAN_model.fit(train_data)
# Clustered_data = DBSCAN_model.fit_predict(train_data)
# Cluster_labels = np.unique(Clustered_data)
# if Show_Plot:
#     fig=plt.figure(figsize=[4,3])
#     ax0=fig.add_subplot(1,1,1)
#     for cluster in Cluster_labels:
#         Cluster_points=train_data[Clustered_data==cluster]
#         ax0.scatter(Cluster_points[:,0],Cluster_points[:,1],label=f'Cluster {cluster}')
#     ax0.grid()
#     ax0.legend()
#     plt.show()

K=4
train_data = pseudoDB[['Vposition_lat','Vposition_long']].values
# Initialize the k-means algorithm with K clusters
kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the k-means algorithm to the data
kmeans.fit(train_data)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Get the coordinates of the cluster centers
Cluster_centers = kmeans.cluster_centers_
Clustered_data = kmeans.transform(train_data).argmin(axis=1) # Classify the point into its cluster
Cluster_labels=np.unique(Clustered_data)

Show_plot=True # Look at the clusters
if Show_plot:
    fig=plt.figure(figsize=[4,3])
    ax0=fig.add_subplot(1,1,1)
    for i in Cluster_labels:
        cluster_points = train_data[Clustered_data == i]
        ax0.scatter(cluster_points[:,0],cluster_points[:,1],label=f'Cluster {i}')
        ax0.scatter(Cluster_centers[i,0],Cluster_centers[i,1],marker='X',label=f'Center for {i}')
    ax0.grid()
    ax0.set_xlabel('Lat')
    ax0.set_ylabel('Lon')
    ax0.set_title('Clusters')
    ax0.legend()
    plt.show()
print(Cluster_labels)
