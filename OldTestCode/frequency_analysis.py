import sys
sys.path.append('../DIVCODE')
from code_functions import *

# TDF0=DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'),Weird_format=True)
TDF0=DF_to_segmented_DF(pd.read_parquet('data/2022-06-05-12-12-09 (1).parquet'))#.iloc[:500])

SDF0=frequency_analysis(TDF0)
fig = plt.figure()
ax=fig.add_subplot(1,1,1)
for kind in np.unique(SDF0['class'].tolist()):
    points=SDF0.loc[SDF0['class']==kind]
    ax.scatter(points['freq'],points['freq_amplitude'],label=kind)
ax.axhline(0.002,c='r')
ax.legend()
ax.grid()
ax.set_xlabel('Frequency')
plt.show()