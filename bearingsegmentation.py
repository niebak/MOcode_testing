import pandas as pd
import numpy as np
from tabulate import tabulate
import sys
sys.path.append('../DIVCODE')
from code_functions import *

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data

# Load data into a dataframe
# df=pd.read_excel(coursename)
ownfile='data/Evening_Run.fit'
variables = ['position_lat', 'position_long', 'altitude','distance']

tdf=fit_records_to_frame(ownfile,variables)
df=tdf[variables].dropna()
# Add a new column for bearing
df['position_lat'] = df['position_lat']/(2**32/360)
df['position_long'] = df['position_long']/(2**32/360)
df['bearing'] = np.nan

#set sensitivity
sensitivity = 15 # change this value to adjust sensitivity
alt_sensitivity = 0 # change this value to adjust sensitivity

# Loop through the dataframe to calculate the bearing at each point
for i in range(1, len(df)):
    lat1 = np.radians(df.loc[i-1, 'position_lat'])
    lat2 = np.radians(df.loc[i, 'position_lat'])
    long1 = np.radians(df.loc[i-1, 'position_long'])
    long2 = np.radians(df.loc[i, 'position_long'])
    df.loc[i, 'bearing'] = np.degrees(np.arctan2(np.sin(long2-long1)*np.cos(lat2), np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(long2-long1)))

# Add new column for incline/decline
df['incline'] = np.nan

# loop through the dataframe to calculate the incline/decline
for i in range(1, len(df)):
    df.loc[i, 'incline'] = df.loc[i, 'altitude'] - df.loc[i-1, 'altitude']

# Add new column for segments
df['segments'] = 0

markers=detect_and_mark_change_in_direction(df['bearing'].tolist(),df['incline'].tolist(),df['segments'])
segments=marker_to_segment(markers)
df['segments'] = segments

# loop through the dataframe to segment the data
# segment_count = 0
# for i in range(1, len(df)):
#     if abs(df.loc[i, 'bearing'] - df.loc[i-1, 'bearing']) > sensitivity:
#         segment_count += 1
#         df.loc[i, 'segments'] = segment_count
#     else:
#         if abs(df.loc[i, 'incline'] - df.loc[i-1, 'incline']) > sensitivity:
#             segment_count += 1
#             df.loc[i, 'segments'] = segment_count
#         else:
#             df.loc[i, 'segments'] = segment_count
print(tabulate(df[['segments','bearing','incline']],headers='keys',tablefmt='github'))
a,b=plot_segments_and_trail(df,Show_plot=True)
