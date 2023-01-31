from code_functions import *

import numpy as np

def add_random_values(df, col1, col2, col3, lower_bound, upper_bound):
    df[col1] += np.random.uniform(lower_bound, upper_bound, df.shape[0])
    df[col2] += np.random.uniform(lower_bound, upper_bound, df.shape[0])
    df[col3] += np.random.uniform(lower_bound, upper_bound, df.shape[0])
    return df

database = 'LUW_example.xlsx'
# database = 'LUWfeature_example.xlsx'
base = 'LUW_data/trail'
columns = ['timestamp','seconds','position_lat','position_long','altitude','segments','distance','velocity [m/s]']
columns2 = ['segments','curvature','revcurvature','climb','revclimb','seg_distance']

trail = pd.read_excel('data/221005_eksempelsegment001.xlsx')
trail=add_random_values(trail,'position_lat','position_long','altitude',0,0.009)
add_to_database(DF_to_segmented_DF(trail),databasename=database,variables=columns)
# for i in range(0,11):
#     FIL = base + str(i) + '.fit'
#     df2 = segments_to_feature_df_with_rev(DF_to_segmented_DF(fit_records_to_frame(FIL,vars=columns)))
#     if 'timestamp' not in columns:
#         columns.append('timestamp')
#     add_to_database(df2,databasename=database,variables=columns2)
