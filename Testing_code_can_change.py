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
    curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,4)
    climb[i]=calculate_height_gained(segment)
    seg_dist[i]=find_distance(segment)
featdict={'curvature':curvature,'climb':climb,'seg_distance':seg_dist}
featureDF=pd.DataFrame(featdict)
print(featureDF)
Show_Plot=True
if Show_Plot:
    ax0,ax1=plot_segments_and_trail(TDF0)
    ax1.legend()
    plt.show()
