import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
from code_functions import *

# defines
Verbose=False
Show_Plot=False

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
# Read data
TDF0=pd.read_excel(coursename)
# TDF0=pd.read_parquet(parquetfile)
Weird_format=True # If the data is from the xlsx file, it is in a weird format so we need to change this
if Weird_format:
    # The data is in a weird format so we need to change it:
    TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
    TDF0['position_long'] = TDF0['position_long']/(2**32/360)
# Verbose = False # Look at the first ten points in the dataframe
if Verbose:
    print(tabulate(TDF0.iloc[0:10],headers='keys',tablefmt='github'))

vector_Coordinates=coordinate_to_vector_dataframe(TDF0) # Change representation

#Show_Plot = True # Look at the course 
if Show_Plot:
    fig = plt.figure()
    ax0=fig.add_subplot(1,1,1,)
    ax0.plot(TDF0['position_lat'],TDF0['position_long'])
    ax0.grid()
    plt.show()

# Used to create the markers
vlat=vector_Coordinates['Vposition_lat'].tolist()
vlon=vector_Coordinates['Vposition_long'].tolist()
valt=vector_Coordinates['Valtitude'].tolist()

segment_marker = detect_and_mark_change_in_direction(vlat,vlon,valt) # Create markers for the segments

segments = marker_to_segment(segment_marker) # Go from markers to segments
TDF0['segments'] = segments
TDF0=pd.concat([TDF0,vector_Coordinates],axis=1) # Add change in the dimensions to the dataframe

Show_Plot=False # Look at the segments
if Show_Plot:
    fig = plt.figure()
    ax0=fig.add_subplot(2,1,1)
    ax0.plot(TDF0['position_long'],TDF0['position_lat'])
    ax0.grid()
    ax1=fig.add_subplot(2,1,2)
    for i in range(segments[0],segments[-1]+2):
        testdf=TDF0.loc[TDF0['segments']==i]
        ax1.plot(testdf['position_long'],testdf['position_lat'])
    plt.show()

Verbose = False # See the length of the segments in points
if Verbose:
    for i in range(segments[0],segments[-1]+2):
        testdf=TDF0.loc[TDF0['segments']==i]
        print(testdf.shape[0])