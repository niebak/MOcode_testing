import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
import sys
sys.path.append('../DIVCODE')
from code_functions import *
from sklearn.cluster import KMeans,DBSCAN
import os
# To limit memory leak, will result in a higher runtime
os.environ["OMP_NUM_THREADS"] = "1"
# defines
Verbose=False
Show_Plot=False

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'
variables = ['position_lat', 'position_long', 'altitude','distance']
TDF=fit_records_to_frame(ownfile,variables)
TDF0=DF_to_segmented_DF(TDF[['position_lat','position_long','altitude']].dropna(),Weird_format=True).iloc[0:300]
print(TDF0.columns)
# Read data
# TDF0= DF_to_segmented_DF(pd.read_excel(coursename),Weird_format=True)
def detect_sections(df, altitude_sensitivity=1):
    df["section"] = "straight" # initialize all sections as straight
    for i in range(1, len(df) - 1):
        # calculate change in position from previous point
        lat_diff = df.loc[i, "position_lat"] - df.loc[i-1, "position_lat"]
        long_diff = df.loc[i, "position_long"] - df.loc[i-1, "position_long"]
        alt_diff = df.loc[i, "altitude"] - df.loc[i-1, "altitude"]
        # check for right or left turn
        if lat_diff * long_diff > 0:
            if lat_diff > 0:
                df.loc[i, "section"] = "right turn"
            else:
                df.loc[i, "section"] = "left turn"
        # check for incline or decline
        elif abs(alt_diff) > altitude_sensitivity:
            if alt_diff > 0:
                df.loc[i, "section"] = "incline"
            else:
                df.loc[i, "section"] = "decline"
    return df

TDF0 = detect_sections(TDF0)

curvature= [0]*(TDF0['segments'].iloc[-1]+1)
climb = [0]*(TDF0['segments'].iloc[-1]+1)
seg_dist = [0]*(TDF0['segments'].iloc[-1]+1)

for i in range(TDF0['segments'].iloc[-1]+1):
    segment = TDF0.loc[TDF0["segments"]==i]
    curvature[i]=calculate_distance_from_straight_line(segment)
    climb[i]=calculate_height_gained(segment)
    # seg_dist[i]=find_distance(segment)
featdict={'curvature':curvature,'climb':climb}
featureDF=pd.DataFrame(featdict)
print(featureDF)
curvature=featureDF['curvature'].tolist()
for i in range(0,TDF0['segments'].iloc[-1]+1):
    if abs(curvature[i])>=0.0001:
        print(i,'turn',round(curvature[i],8))
    else:
        print(i,'straight',round(curvature[i],4))
ax0,ax1=plot_segments_and_trail(TDF0)
ax1.legend()
print(np.mean(curvature))
plt.show()