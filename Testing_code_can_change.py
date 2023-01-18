import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN
import sys
sys.path.append('../DIVCODE')
from code_functions import *
import os

# To limit memory leak, will result in a higher runtime
os.environ["OMP_NUM_THREADS"] = "1"


# defines
Verbose=False
Show_Plot=False
variables = ['position_lat', 'position_long', 'altitude']

coursename = "data/221005_eksempelsegment001.xlsx" # Short, but with a lot of data
parquetfile='data/2022-06-05-12-12-09 (1).parquet' # Longer, from strava
ownfile='data/Evening_Run.fit'

TDF0=pd.read_excel(coursename)
TDF1=pd.read_parquet(parquetfile)
TDF=fit_records_to_frame(ownfile,variables)
TDF2=TDF[['position_lat','position_long','altitude']].dropna()
#  Read data

TDF0 = DF_to_segmented_DF(TDF0)
TDF1 = DF_to_segmented_DF(TDF1)
TDF2 = DF_to_segmented_DF(TDF2)

