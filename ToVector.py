# %% Import, define, and read
# Imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tabulate import tabulate
from code_functions import *

# Some meta stuff
show_plots=True
verbose=False

# Defines
coursename = "data/221005_eksempelsegment001.xlsx"
parquetfile='data/2022-06-05-12-12-09 (1).parquet'
# Read data
TDF0=pd.read_excel(coursename)

# The data is in a weird format so we need to change it:
TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
TDF0['position_long'] = TDF0['position_long']/(2**32/360)
# look at the data
if False:
    fig=plt.figure()
    ax0=fig.add_subplot(1,1,1)
    ax0.plot(TDF0['position_lat'],TDF0['position_long'])
    ax0.set_title('A quick peek at the raw data')
    ax0.grid()
    plt.show()
# %% Creating a vector with the same information 
coordinates=TDF0[["position_lat","position_long","altitude"]]
VectorCoordinates=[coordinates.iloc[0].tolist()]*coordinates.shape[0]
for point in range(0,coordinates.shape[0]-1):
    VectorCoordinates[point+1]=coordinate_to_vector(coordinates.iloc[point].tolist(),coordinates.iloc[point+1].tolist())
VectorDataframe=pd.DataFrame(VectorCoordinates,columns=['position_lat',"position_long","altitude"])
print(f'shape of {VectorDataframe.shape}.')
# %% We have now verified that the vector transform works, and next is to try to create some segments based upon this
SegmentMarkers=cumulative_sum_with_limit(
    VectorDataframe['position_lat'],
    VectorDataframe['position_long'],
    VectorDataframe['altitude'],
    l1=0.003,
    l2=0.003,
    l3=5)
segments=marker_to_segment(SegmentMarkers)
print(f'{segments[-1]} segments')
if verbose:
    print(segments)
    print(SegmentMarkers)
TDF0['segments']=segments
# %% Take a look at the created segments:
if show_plots:
    fig1 = plt.figure(figsize=[6,2])
    ax1 = plt.axes(projection='3d')
    for i in range(0,segments[-1]+1):
        testdf=TDF0.loc[TDF0['segments']==i]
        ax1.plot(testdf['position_long'],testdf['position_lat'],testdf['altitude'])
    plt.show()