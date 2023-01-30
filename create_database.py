from code_functions import *


database = 'LUW_example.xlsx'
base = 'LUW_data/trail'
columns = ['timestamp','seconds','position_lat','position_long','altitude','segments','distance','velocity [m/s]']
add_to_database(DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx')),databasename=database,variables=columns)
# for i in range(0,11):
#     FIL = base + str(i) + '.fit'
#     df2 = DF_to_segmented_DF(fit_records_to_frame(FIL,vars=columns))
#     if 'timestamp' not in columns:
#         columns.append('timestamp')
#     add_to_database(df2,databasename=database,variables=columns)