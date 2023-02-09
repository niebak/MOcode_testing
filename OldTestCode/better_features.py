from code_functions import DF_to_segmented_DF,print_df,calculate_distance_from_straight_line
from code_functions import calculate_height_gained,find_distance
from code_functions import plot_segments_and_trail
import pandas as pd
import numpy as np

trailname = 'data/221005_eksempelsegment001.xlsx'

TDF0 = DF_to_segmented_DF(pd.read_excel(trailname))
print(TDF0.columns)
plot_segments_and_trail(TDF0, Show_Plot=True)
curvature= [0]*(TDF0['segments'].iloc[-1]+1)
climb = [0]*(TDF0['segments'].iloc[-1]+1)
seg_dist = [0]*(TDF0['segments'].iloc[-1]+1)
grade = [0]*(TDF0['segments'].iloc[-1]+1)
velturn = [0]*(TDF0['segments'].iloc[-1]+1)
segments = TDF0['segments'].tolist()
for i in range(TDF0['segments'].iloc[-1]+1): # Loop through each segment
    segment = TDF0.loc[TDF0["segments"]==i]
    if segment.shape[0]==0:
        print(f'problem at index {i}, do some troubleshooting.')
    else:
        # find the curvature as a distance from a straight line
        curvature[i]=round(calculate_distance_from_straight_line(segment)*10**4,1)  
        if abs(curvature[i])<1:
            curvature[i]=0
        # Find the altitude difference
        climb[i]=calculate_height_gained(segment)
        if abs(climb[i])<1:
            climb[i]=0
        # find the distance
        seg_dist[i]=find_distance(segment)
        grade[i] = round((climb[i]/ seg_dist[i])*100,1)
        velturn[i]= curvature[i]*abs(TDF0['velocity [m/s]'].loc[TDF0['segments']==i].mean())/10
featdict={'segments':np.unique(segments),'curvature':curvature,'climb':climb,'seg_distance':seg_dist,'grade':grade,'velturn':velturn}
featureDF=pd.DataFrame(featdict)
print_df(featureDF)