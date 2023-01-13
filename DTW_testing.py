import pandas as pd
# from dtw import dtw
import pyarrow as pa
import pyarrow.parquet as pq

coursename = "data/221005_eksempelsegment001.xlsx"
parquetfile='data/2022-06-05-12-12-09 (1).parquet'

trail1=pd.read_csv(coursename)
trail2=pd.read_parquet(parquetfile)
print(trail1,trail2)
