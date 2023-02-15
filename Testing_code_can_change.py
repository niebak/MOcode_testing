import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
import sys
sys.path.append('../DIVCODE')
from code_functions import *
from sklearn.cluster import KMeans,DBSCAN
import math
import os
# To limit memory leak, will result in a higher runtime when doing Kmeans
os.environ["OMP_NUM_THREADS"] = "1"

file = 'data/Evening_Run.fit'
earth_radius = 6371e3  # radius of Earth in meters

TDF0 = DF_to_segmented_DF(fit_records_to_frame(file))

for i in (np.unique(TDF0['segments'])):
    wdf = TDF0.loc[TDF0['segments']==i].copy(deep=True)
    total_distance = round(sum(wdf['diff_distance']),2)
    sum_diff_lat = sum(wdf['Vposition_lat'])
    sum_diff_long = sum(wdf['Vposition_long'])

    sum_diff_lat_m = sum_diff_lat * math.pi / 180 * earth_radius * math.cos(wdf['position_lat'].iloc[0] * math.pi / 180)
    sum_diff_long_m = sum_diff_long * math.pi / 180 * earth_radius
    
    stldist = np.sqrt(sum_diff_long_m**2+sum_diff_lat_m**2)


    print('\n')
    print(i)
    print(sum_diff_lat_m,sum_diff_long_m)
    print(total_distance/stldist)
    print('\n')
    ax0,ax1=plot_segments_and_trail(wdf)
    ax1.set_xlabel(sum(wdf['Valtitude']))
    plt.show()