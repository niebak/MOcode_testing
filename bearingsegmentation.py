# %%
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN
import sys
sys.path.append('../DIVCODE')
from code_functions import *
import os

Show_Plot=False

# To limit memory leak, will result in a higher runtime
os.environ["OMP_NUM_THREADS"] = "1"


coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data

# Load data into a dataframe
# df=pd.read_excel(coursename)
ownfile='data/Evening_Run.fit'
variables = ['position_lat', 'position_long', 'altitude','distance']

# tdf=fit_records_to_frame(ownfile,variables)
# df=tdf[variables].dropna()
df=pd.read_excel(coursename)
# weird format
df['position_lat'] = df['position_lat']/(2**32/360)
df['position_long'] = df['position_long']/(2**32/360)
# Add a new column for bearing
# df['bearing'] = np.nan

#set sensitivity
sensitivity = 15 # change this value to adjust sensitivity
alt_sensitivity = 5 # change this value to adjust sensitivity

# Loop through the dataframe to calculate the bearing at each point
for i in range(1, len(df)):
    lat1 = np.radians(df.loc[i-1, 'position_lat'])
    lat2 = np.radians(df.loc[i, 'position_lat'])
    long1 = np.radians(df.loc[i-1, 'position_long'])
    long2 = np.radians(df.loc[i, 'position_long'])
    df.loc[i, 'bearing'] = np.degrees(np.arctan2(np.sin(long2-long1)*np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(long2-long1)))

# Add new column for incline/decline
df['incline'] = np.nan

# loop through the dataframe to calculate the incline/decline
for i in range(1, len(df)):
    df.loc[i, 'incline'] = df.loc[i, 'altitude'] - df.loc[i-1, 'altitude']

# Add new column for segments
df['segments'] = 0

# Create the segments
markers=detect_and_mark_change_in_direction(df['bearing'].tolist(),df['segments'],df['altitude'].tolist())
segments=marker_to_segment(markers)
df['segments'] = segments
print(f' number of segments {segments[-1]}')
sdf=create_segmentDF_fromDF(df,variables=['segments','bearing','incline','altitude'])


c0=[0]*(df['segments'].iloc[-1]+1)
l0=[0]*(df['segments'].iloc[-1]+1)
a0=[0]*(df['segments'].iloc[-1]+1)

variables = ['position_lat', 'position_long', 'altitude','distance']
for i in range(0,df['segments'].iloc[-1]):
    wdf = wdf=df[variables].loc[df['segments']==i]
    c0[i]=(calculate_curvature(wdf))
    l0[i]=(find_distance(wdf))
    a0[i]=calculate_height_gained(wdf)

sdf['curvature']=c0
sdf['distance']=l0
sdf['hill']=a0
sdf['segments']=range(0,df['segments'].iloc[-1]+1)


print(tabulate(sdf,headers='keys',tablefmt='github'))


if Show_Plot:
    fig = plt.figure()
    ax0=fig.add_subplot(1,1,1)
    ax0.scatter(sdf['curvature'],sdf['incline'])
    ax0.grid()
    plt.show()
K=5
train_data = sdf.values
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

Show_Plot=False # Look at the clusters
if Show_Plot:
    fig=plt.figure(figsize=[4,3])
    ax0=fig.add_subplot(1,1,1)
    for i in Cluster_labels:
        if i ==0:
            colour='r'
        
        if i ==1:
            colour='g'

        if i ==2:
            colour='b'

        if i ==3:
            colour='y'

        if i ==4:
            colour='c'
        
        cluster_points = train_data[Clustered_data == i]
        ax0.scatter(cluster_points[:,0],cluster_points[:,1],color=colour,label=f'Cluster {i}')
        ax0.scatter(Cluster_centers[i,0],Cluster_centers[i,1],color=colour,marker='X',label=f'Center for {i}')
    ax0.grid()
    ax0.set_xlabel('curvature')
    ax0.set_ylabel('hill')
    ax0.set_title('Clusters')
    ax0.legend()
    plt.show()
# %%
Show_Plot = True

clusterlist = [0]*(len(segments))
for i in range(0,len(clusterlist)-1):
    clusterlist[i]=Clustered_data[segments[i]]
df['clusters']=clusterlist
sdf['clusters']=Clustered_data
print(tabulate(sdf[['segments','incline','curvature','clusters']],headers='keys',tablefmt='github'))
if Show_Plot:
    fig=plt.figure(figsize=[4,3])
    ax0=fig.add_subplot(1,1,1)
    for i in segments:
        wdf = df[['position_lat', 'position_long', 'altitude','distance','clusters']].loc[df['segments']==i]
        seg_cluster=wdf['clusters'].iloc[3]
        if seg_cluster ==0:
            colour='r'
        if seg_cluster ==1:
            colour='g'
        if seg_cluster ==2:
            colour='b'
        if seg_cluster ==3:
            colour='y'
        if seg_cluster ==4:
            colour='c'
        ax0.plot(wdf['position_lat'],wdf['position_long'],color=colour,label=f'Cluster {i}')
    ax0.grid()
    ax0.set_xlabel('Lat')
    ax0.set_ylabel('Lon')
    ax0.set_title('Looking at the clusters')
    # ax0.legend()
    plt.show()

# %%
