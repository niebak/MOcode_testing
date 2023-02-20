from code_functions import *
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB = pd.read_excel('LUW_example.xlsx')
stats = DB.describe()
Candidate = DB.loc[DB['name']==1].copy(deep=True)
# plot_segments_and_trail(Candidate,Show_Plot=False)
lengths = []
# for i in np.unique(Candidate['segments']):
meanlat = np.mean(Candidate['position_lat'].to_numpy())
meanlong = np.mean(Candidate['position_long'].to_numpy())
xs=[]
ys=[]
for i in np.unique(Candidate['segments'].to_numpy()):
    segment = Candidate.loc[Candidate['segments']==i].copy(deep=True)
    # print_df(segment)
    # fig=plt.figure()
    # ax0=fig.add_subplot(2,1,1)
    # ax1=fig.add_subplot(2,1,2)

    # ax0.plot(segment['position_long'],segment['position_lat'])
    start_x,start_y = segment['position_long'].iloc[0],segment['position_lat'].iloc[0]
    stop_x,stop_y = segment['position_long'].iloc[-1],segment['position_lat'].iloc[-1]
    
    x=[start_x,stop_x]
    y=[start_y,stop_y]
    distance = (segment['distance'].diff().to_numpy())
    distance[0]=0
    distance=sum(distance)
    xs.append(np.mean(x))
    ys.append(np.mean(y))
    # ax1.plot(x,y,label = distance)
    # ax1.legend()
    # print('\n',(np.mean(x)-meanlong)*10**6,(np.mean(y)-meanlat)*10**6)
    # ax0.scatter(np.mean(x),np.mean(y))
    # print('\n',np.mean(x),np.mean(y))
    # print(np.std(x),np.std(y),'\n')

    # plt.show()
fig1 = plt.figure()
ax=fig1.add_subplot(1,1,1)
ax.scatter(xs,ys)
ax.grid()
plt.show()