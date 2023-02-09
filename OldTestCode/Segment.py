import numpy as np
import matplotlib as plt
from tabulate import tabulate 
import pandas as pd
from code_functions import *

# defines
Verbose=True
Show_Plot=False

Weird_format=True

coursename = "data/221005_eksempelsegment001.xlsx"
parquetfile='data/2022-06-05-12-12-09 (1).parquet'
# Read data
TDF0=pd.read_excel(coursename)
if Weird_format:
    # The data is in a weird format so we need to change it:
    TDF0['position_lat'] = TDF0['position_lat']/(2**32/360)
    TDF0['position_long'] = TDF0['position_long']/(2**32/360)
if Verbose:
    print(tabulate(TDF0,headers='keys',tablefmt='github'))

vector_Coordinates=coordinate_to_vector_dataframe(TDF0)

segment_marker = cumulative_sum_with_limit(vector_Coordinates['Vposition_lat'],vector_Coordinates['Vposition_long'],vector_Coordinates['Valtitude'])
segments = marker_to_segment(segment_marker)

TDF0['segments'] = segments
TDF0=pd.concat([TDF0,vector_Coordinates],axis=1)
plot_segments_and_trail(TDF0,Show_Plot=True)

