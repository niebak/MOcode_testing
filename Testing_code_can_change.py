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
Show_Plot=True

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'
variables = ['position_lat', 'position_long', 'altitude','distance']
# TDF=fit_records_to_frame(ownfile,variables)
# TDF0=DF_to_segmented_DF(TDF[['position_lat','position_long','altitude']].dropna()
# ,Weird_format=True)#.iloc[0:300]
# TDF0 = DF_to_segmented_DF(pd.read_excel(coursename),Weird_format=True)
TDF0=DF_to_segmented_DF(pd.read_parquet(parquetfile)).iloc[0:200]

# Read data
curvature= [0]*(TDF0['segments'].iloc[-1]+1)
climb = [0]*(TDF0['segments'].iloc[-1]+1)
seg_dist = [0]*(TDF0['segments'].iloc[-1]+1)

for i in range(TDF0['segments'].iloc[-1]+1):
    segment = TDF0.loc[TDF0["segments"]==i]
    curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,1)
    if abs(curvature[i])<1:
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
# Trying to write a classifier
Segment_Class=['']*featureDF.shape[0]

curve_lim=1
climb_lim=3

for segment in range(0,featureDF.shape[0]):
    curve=featureDF['curvature'].iloc[segment]
    climb=featureDF['climb'].iloc[segment]
    distance=featureDF['seg_distance'].iloc[segment]
    if curve> curve_lim:
        Segment_Class[segment] = 'R turn'
    if abs(curve)<=curve_lim:
        Segment_Class[segment]='straight'
    if curve<-curve_lim:
        Segment_Class[segment]='L turn'
    if climb>=climb_lim:
        Segment_Class[segment]+=(' incline')
    if climb<=-climb_lim:
        Segment_Class[segment]+=(' decline')
    if Segment_Class[segment] == '':
        Segment_Class[segment] = 'Unknown class'
featureDF['class']=Segment_Class
# print(Segment_Class)
# print(np.unique(Segment_Class),len(np.unique(Segment_Class)))
print('\n\n\n')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for kind in np.unique(featureDF['class'].tolist()):
    points=featureDF.loc[featureDF['class']==kind]
    ax.scatter(points['curvature'],points['climb'],label=kind)
ax.legend()
ax.grid()
ax.set_xlabel('Curvature')
ax.set_ylabel('Climb')
plt.show()

print(tabulate(featureDF,headers='keys',tablefmt='github'))
print(featureDF['class'].value_counts())