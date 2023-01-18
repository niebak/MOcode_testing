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

TDF0_seg =  marker_to_segment(TDF0_seg_marker)
TDF0=pd.concat([TDF0,TDF0_vector],axis=1)
TDF0['segments']=TDF0_seg


TDF1_seg =  marker_to_segment(TDF1_seg_marker)
TDF1=pd.concat([TDF1,TDF1_vector],axis=1)
TDF1['segments']=TDF1_seg

TDF2_seg =  marker_to_segment(TDF2_seg_marker)
TDF2=pd.concat([TDF2,TDF2_vector],axis=1)
TDF2['segments']=TDF2_seg

print(TDF0_seg[-1],TDF1_seg[-1],TDF2_seg[-1])