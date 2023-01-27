
import pandas as pd

from dtw import dtw
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
print('testtest')
coursename = "data/221005_eksempelsegment001.xlsx"
parquetfile='data/2022-06-05-12-12-09 (1).parquet'

Columns=['position_lat','position_long','altitude']

# Read data
TDF0=pd.read_excel(coursename)[Columns]
print(type(TDF0.values))
TDF1=pd.read_parquet(parquetfile)[Columns]
# print(TDF1.values)
distance = dtw(TDF0.values, TDF1.values)

# distance.plot()

# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# ax.plot(distance.index1,distance.index2)
# plt.show()

# ldist = np.ones((6,6))                    # Matrix of ones
# ldist[1,:] = 0; ldist[:,4] = 0           # Mark a clear path of zeroes
# ldist[1,4] = .01

# ds = dtw(ldist)
# ax.plot(ds.index1,ds.index2)
# plt.show()