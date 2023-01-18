import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
import sys
sys.path.append('../DIVCODE')
from code_functions import *
import os
from math import sin, cos, sqrt, atan2

# To limit memory leak, will result in a higher runtime
os.environ["OMP_NUM_THREADS"] = "1"


# defines
Verbose=False
Show_Plot=False
variables = ['position_lat', 'position_long', 'altitude','distance']

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'

def calculate_curvature(dataframe):
    x = dataframe['position_long'].values
    y = dataframe['position_lat'].values
    dx = np.diff(x)
    dy = np.diff(y)
    curvature = np.zeros(len(dx))
    for i in range(len(dx)):
        if dx[i] == 0 or dy[i] == 0:
            curvature[i] = 0
        else:
            curvature[i] = (dy[i] / dx[i]) / (1 + (dy[i] / dx[i])**2)**(3/2)
    return curvature
def find_distance(df):
    '''
    Works on segments. Assumes distance is present in the data
    '''
    start=df['distance'].iloc[0]
    stop=df['distance'].iloc[-1]
    return round(stop-start,2)
def hav_distance(df):
    R = 6378
    lat1 = df['position_lat'].apply('radians')
    lat2 = df['position_lat'].shift(-1).apply('radians')
    lon1 = df['position_long'].apply('radians')
    lon2 = df['position_long'].shift(-1).apply('radians')
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance.sum()
def calculate_height_gained(dataframe):
    start = dataframe['altitude'].iloc[0]
    stop = dataframe['altitude'].iloc[-1]
    height_gained = stop - start
    return round(height_gained,2)



TDF0=pd.read_excel(coursename)
TDF1=pd.read_parquet(parquetfile)
TDF=fit_records_to_frame(ownfile,variables)
TDF2=TDF[variables].dropna()
#  Read data
print(TDF0.columns)
TDF0 = DF_to_segmented_DF(TDF0)
TDF1 = DF_to_segmented_DF(TDF1)
TDF2 = DF_to_segmented_DF(TDF2)

c0=[0]*TDF0['segments'].iloc[-1]
l0=[0]*TDF0['segments'].iloc[-1]
a0=[0]*TDF0['segments'].iloc[-1]


for i in range(0,TDF0['segments'].iloc[-1]):
    wdf = wdf=TDF0[variables].loc[TDF0['segments']==i]
    c0[i]=sum(calculate_curvature(wdf))
    l0[i]=(find_distance(wdf))
    a0[i]=calculate_height_gained(wdf)
SDF0=pd.DataFrame()
SDF0['curvature']=c0
SDF0['distance']=l0
SDF0['hill']=a0
SDF0['segments']=range(0,TDF0['segments'].iloc[-1])

c1=[0]*TDF1['segments'].iloc[-1]
l1=[0]*TDF1['segments'].iloc[-1]
a1=[0]*TDF1['segments'].iloc[-1]


for i in range(0,TDF1['segments'].iloc[-1]):
    wdf = wdf=TDF1[variables].loc[TDF1['segments']==i]
    c1[i]=sum(calculate_curvature(wdf))
    l1[i]=(find_distance(wdf))
    a1[i]=calculate_height_gained(wdf)
SDF1=pd.DataFrame()
SDF1['curvature']=c1
SDF1['distance']=l1
SDF1['hill']=a1
SDF1['segments']=range(0,TDF1['segments'].iloc[-1])

c2=[0]*TDF2['segments'].iloc[-1]
l2=[0]*TDF2['segments'].iloc[-1]
a2=[0]*TDF2['segments'].iloc[-1]


for i in range(0,TDF2['segments'].iloc[-1]):
    wdf = wdf=TDF2[variables].loc[TDF2['segments']==i]
    c2[i]=sum(calculate_curvature(wdf))
    l2[i]=(find_distance(wdf))
    a2[i]=calculate_height_gained(wdf)
SDF2=pd.DataFrame()
SDF2['curvature']=c2
SDF2['distance']=l2
SDF2['hill']=a2
SDF2['segments']=range(0,TDF2['segments'].iloc[-1])
