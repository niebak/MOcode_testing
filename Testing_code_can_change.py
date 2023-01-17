import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
import sys
sys.path.append('../DIVCODE')
from code_functions import *

# defines
Verbose=False
Show_Plot=False
variables = ['position_lat', 'position_long', 'altitude', 'distance',
'enhanced_altitude']

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'

TDF=fit_records_to_frame(ownfile,variables)
# TDF.drop(TDF.index[:200], inplace=True)
TDF2=TDF[['position_lat','position_long','altitude']].dropna()

# TDF2=TDF2
# Read data
TDF0=pd.read_excel(coursename)
#TDF1=pd.read_parquet(parquetfile)
Weird_format=True # If the data is from the xlsx file, it is in a weird format so we need to change this
if Weird_format:
    # The data is in a weird format so we need to change it:
    TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
    TDF0['position_long'] = TDF0['position_long']/(2**32/360)
    TDF2['position_lat'] = TDF2['position_lat']/(2**32/360)
    TDF2['position_long'] = TDF2['position_long']/(2**32/360)

# Create the vector-representation coordinates
TDF0_vector = coordinate_to_vector_dataframe(TDF0)
TDF2_vector = coordinate_to_vector_dataframe(TDF2)
print(len(TDF2_vector))
#TDF1_vector = coordinate_to_vector_dataframe(TDF1)
TDF0_seg_marker=detect_and_mark_change_in_direction(
    TDF0_vector['Vposition_lat'].tolist(),
    TDF0_vector['Vposition_long'].tolist(),
    TDF0_vector['Valtitude'].tolist())

TDF2_seg_marker=detect_and_mark_change_in_direction(
    TDF2_vector['Vposition_lat'].tolist(),
    TDF2_vector['Vposition_long'].tolist(),
    TDF2_vector['Valtitude'].tolist())

TDF0_seg =  marker_to_segment(TDF0_seg_marker)
TDF0=pd.concat([TDF0,TDF0_vector],axis=1)
TDF0['segments']=TDF0_seg

TDF2_seg =  marker_to_segment(TDF2_seg_marker)
print(len(TDF2_vector))
TDF2=pd.concat([TDF2,TDF2_vector],axis=1)
# print(len(TDF2))

TDF2['segments']=TDF2_seg

plot_segments_and_trail(TDF2,Show_plot=True)
for segment in np.unique(TDF0['segments'].tolist()):
    wdf=TDF2[['Vposition_lat','Vposition_long','Valtitude']].loc[TDF2['segments']==segment]
    print(f'\nSegment {segment} '+str(wdf.describe().loc[['mean','std','count']]))