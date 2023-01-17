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

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava

# Read data
TDF0=pd.read_excel(coursename)
TDF1=pd.read_parquet(parquetfile)
Weird_format=True # If the data is from the xlsx file, it is in a weird format so we need to change this
if Weird_format:
    # The data is in a weird format so we need to change it:
    TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
    TDF0['position_long'] = TDF0['position_long']/(2**32/360)

# Create the vector-representation coordinates
TDF0_vector = coordinate_to_vector_dataframe(TDF0)
TDF1_vector = coordinate_to_vector_dataframe(TDF1)

TDF0_seg_marker=detect_and_mark_change_in_direction(
    TDF0_vector['Vposition_lat'].tolist(),
    TDF0_vector['Vposition_long'].tolist(),
    TDF0_vector['Valtitude'].tolist())

TDF1_seg_marker=detect_and_mark_change_in_direction(
    TDF1_vector['Vposition_lat'].tolist(),
    TDF1_vector['Vposition_long'].tolist(),
    TDF1_vector['Valtitude'].tolist())

TDF0_seg =  marker_to_segment(TDF0_seg_marker)
TDF1_seg =  marker_to_segment(TDF1_seg_marker)

TDF0['segments']=TDF0_seg
TDF1['segments']=TDF1_seg
print(TDF0_seg[-1],TDF1_seg[-1])

ax00,ax01 = plot_segments_and_trail(TDF0)
ax10,ax11 = plot_segments_and_trail(TDF1)
plt.show()
