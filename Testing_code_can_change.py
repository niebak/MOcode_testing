import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate 
import pandas as pd
import sys
sys.path.append('../DIVCODE')
from code_functions import *
from sklearn.cluster import KMeans,DBSCAN
import os
# To limit memory leak, will result in a higher runtime when doing Kmeans
os.environ["OMP_NUM_THREADS"] = "1"
# defines
Verbose=False
Show_Plot=True

DataBase_name = 'data/BTDNB'
DataBase_file = DataBase_name + ('.xlsx')
DataBase_note = DataBase_name + ('.txt')

trailfile = 'data/2022-06-05-12-12-09 (1).parquet'
trailfile1 = 'data/221005_eksempelsegment001.xlsx'
trailfile2= 'data/Evening_Run.fit'
trailfile3 = 'data/Klassisk.fit'

columns = ['timestamp','seconds','position_lat','position_long','altitude','segments','velocity [m/s]']
df0 = DF_to_segmented_DF(pd.read_parquet(trailfile))
df1 = DF_to_segmented_DF(pd.read_excel(trailfile1))
df2 = DF_to_segmented_DF(fit_records_to_frame(trailfile2,vars=columns))
print(df2.columns)
# # df3 = DF_to_segmented_DF(fit_records_to_frame(trailfile3))

add_to_database(df0,databasename=DataBase_file,variables=columns)
add_to_database(df1,databasename=DataBase_file,variables=columns)
add_to_database(df2,databasename=DataBase_file,variables=columns)