import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
import sys
sys.path.append('../DIVCODE')
from code_functions import *
from sklearn.cluster import KMeans,DBSCAN
import os
# To limit memory leak, will result in a higher runtime when doing Kmeans
os.environ["OMP_NUM_THREADS"] = "1"
# defines
Verbose=False
Show_Plot=False

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'
variables = ['position_lat', 'position_long', 'altitude','distance']
TDF=fit_records_to_frame(ownfile,variables)
TDF0=DF_to_segmented_DF(TDF[['position_lat','position_long','altitude']].dropna()
,Weird_format=True).iloc[0:300]
# TDF0 = DF_to_segmented_DF(pd.read_excel(coursename),Weird_format=True)

# Read data
curvature= [0]*(TDF0['segments'].iloc[-1]+1)
climb = [0]*(TDF0['segments'].iloc[-1]+1)
seg_dist = [0]*(TDF0['segments'].iloc[-1]+1)

for i in range(TDF0['segments'].iloc[-1]+1):
    segment = TDF0.loc[TDF0["segments"]==i]
    curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,1)
    if curvature[i]<1:
        curvature[i]=0
    climb[i]=calculate_height_gained(segment)
    if abs(climb[i])<1:
        climb[i]=0
    seg_dist[i]=find_distance(segment)
featdict={'curvature':curvature,'climb':climb,'seg_distance':seg_dist}
featureDF=pd.DataFrame(featdict)
print(featureDF)
# Show_Plot=False
if Show_Plot:
    ax0,ax1=plot_segments_and_trail(TDF0)
    ax1.legend()
    plt.show()


K=4
train_data = featureDF[['curvature','climb']].values
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

# Show_plot=False # Look at the clusters
if Show_Plot:
    fig=plt.figure(figsize=[4,3])
    ax0=fig.add_subplot(2,1,2)
    ax1=fig.add_subplot(2,1,1)
    for i in Cluster_labels:
        cluster_points = train_data[Clustered_data == i]
        ax0.scatter(cluster_points[:,0],cluster_points[:,1],label=f'Cluster {i}')
        ax0.scatter(Cluster_centers[i,0],Cluster_centers[i,1],marker='X',label=f'Center for {i}')
        ax1.scatter(Cluster_centers[i,0],Cluster_centers[i,1],marker='X',label=f'Center for {i}')
    ax0.grid()
    ax1.grid()
    ax1.legend()
    xlim = ax0.get_xlim()
    ylim = ax0.get_ylim()
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax0.set_xlabel('Curvature')
    ax0.set_ylabel('Climb')
    ax0.set_title('Clusters')
    ax0.legend()
    plt.show()
print(Cluster_labels)

# Show_Plot = False

Segment_Class=['']*featureDF.shape[0]

curve_lim=1
climb_lim=3

for segment in range(0,featureDF.shape[0]):
    curve=featureDF['curvature'].iloc[segment]
    climb=featureDF['climb'].iloc[segment]
    distance=featureDF['seg_distance'].iloc[segment]
    if curve<=curve_lim:
        Segment_Class[segment]='straight'
    else:
        Segment_Class[segment]='turn'
    if climb>=climb_lim:
        Segment_Class[segment]+=(' incline')
    if climb<=-climb_lim:
        Segment_Class[segment]+=(' decline')

segments = TDF0['segments'].tolist()
clusterlist = [0]*(len(segments))
for i in range(0,len(clusterlist)-1):
    clusterlist[i]=Segment_Class[segments[i]]
TDF0['clusters']=clusterlist
Show_Plot=True
if Show_Plot:
    fig=plt.figure(figsize=[4,3])
    ax0=fig.add_subplot(1,1,1,projection='3d')
    for i in segments:
        wdf = TDF0[['position_lat', 'position_long', 'altitude','distance','clusters']].loc[TDF0['segments']==i]
        seg_cluster=wdf['clusters'].iloc[0]
        if seg_cluster =='straight':
            colour='r'
        if seg_cluster =='turn':
            colour='g'
        if seg_cluster =='turn incline':
            colour='b'
        if seg_cluster =='turn decline':
            colour='y'
        if seg_cluster ==4:
            colour='c'
        ax0.plot(wdf['position_lat'],wdf['position_long'],wdf['altitude'],color=colour,label=f'Cluster {i}')
    ax0.grid()
    ax0.set_xlabel('Lat')
    ax0.set_ylabel('Lon')
    ax0.set_zlabel('Alt')
    # ax0.set_zlim(400,450)
    ax0.set_title('Looking at the clusters')
    # ax0.legend()
    plt.show()
# Trying to write a classifier
Class_labels = ['straight','straight incline','straight decline',
                'L turn',' L turn incline','L turn decline',
                'R turn','R turn incline','L turn decline']
Segment_Class=['']*featureDF.shape[0]

curve_lim=1
climb_lim=3

for segment in range(0,featureDF.shape[0]):
    curve=featureDF['curvature'].iloc[segment]
    climb=featureDF['climb'].iloc[segment]
    distance=featureDF['seg_distance'].iloc[segment]
    if curve<=curve_lim:
        Segment_Class[segment]='straight'
    else:
        Segment_Class[segment]='turn'
    if climb>=climb_lim:
        Segment_Class[segment]+=(' incline')
    if climb<=-climb_lim:
        Segment_Class[segment]+=(' decline')
print(Segment_Class)
print(np.unique(Segment_Class),len(np.unique(Segment_Class)))