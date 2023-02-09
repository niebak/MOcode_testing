#%%
from code_functions import *
import numpy as np
import pandas as pd
import stumpy
import matplotlib.pyplot as plt

# Load dataframe
df = pd.read_excel('data/221005_eksempelsegment001.xlsx')

# Extract the column of interest
column = df['speed'].values

# Compute the matrix profile
mp = stumpy.stump(column, 64)
print(mp)
print(mp.shape)
profile = mp[:,0]
indices = mp[:,1]
# Identify repeating patterns (motifs) by finding the lowest values in the matrix profile
motifs = np.where(mp == np.min(mp))

# Identify unusual patterns (discords) by finding the highest values in the matrix profile
discords = np.where(mp == np.max(mp))

print(motifs)
print(discords)

#%% 
fig = plt.figure()
ax0=fig.add_subplot(1,1,1)
x = np.linspace(0,len(motifs),len(motifs))
ax0.plot(x,motifs)
ax0.grid()
plt.show()
# %%
print(motifs[0])
# %%
plot_segments_and_trail(DF_to_segmented_DF(df,Weird_format=True),Show_Plot=True)
# %%
