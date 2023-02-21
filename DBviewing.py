from code_functions import *
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB = pd.read_excel('LUW_example.xlsx')
stats = DB.describe()
Candidate = DB.loc[DB['name']==0].copy(deep=True)

def check_for_duplicate_segments(list1,val1,list2,val2,lim=5):
    '''
    Returns Boolean stating if a value is within a limit of a list.
    Works on two lists at a time.
    '''
    # dupe = False
    for i in range(len(list1)):
        element1 = list1[i]
        element2 = list2[i]
        
        diff1 = abs(abs(element1)-abs(val1))
        diff2 = abs(abs(element2)-abs(val2))
        # print(diff1,diff2)

        if (diff1<=lim)and(diff2<=lim):
            print(diff1,diff2)
            return True
    return False

# plot_segments_and_trail(Candidate,Show_Plot=False)
lengths = []
# for i in np.unique(Candidate['segments']):
meanlat = np.mean(Candidate['position_lat'].to_numpy())
meanlong = np.mean(Candidate['position_long'].to_numpy())
xs=[]
ys=[]
# np.unique(Candidate['segments'].to_numpy())
Candidate_without_dupes=pd.DataFrame()
print(xs,ys)
for i in np.unique(Candidate['segments'].to_numpy()):
    segment = Candidate.loc[Candidate['segments']==i].copy(deep=True)

    start_x,start_y = segment['position_long'].iloc[0],segment['position_lat'].iloc[0]
    stop_x,stop_y = segment['position_long'].iloc[-1],segment['position_lat'].iloc[-1]
    
    x=[start_x,stop_x]
    y=[start_y,stop_y]
    distance = (segment['distance'].diff().to_numpy())
    distance[0]=0
    distance=sum(distance)

    scaled_x=(np.mean(x)-meanlong)*10**6
    scaled_y=(np.mean(y)-meanlat)*10**6

    # print(scaled_x,scaled_y)
    
    dupeflag = check_for_duplicate_segments(xs,scaled_x,ys,scaled_y,lim=15)
    if(dupeflag):
        print(f'dupe at {i}/{Candidate["segments"].iloc[-1]}')
    else:
        xs.append(scaled_x)
        ys.append(scaled_y)
        # print('\n',i,scaled_x,scaled_y)
        Candidate_without_dupes=pd.concat([Candidate_without_dupes,segment],ignore_index=True)
# print(xs)
# print(ys)
# fig1 = plt.figure()
# ax=fig1.add_subplot(1,1,1)
# ax.scatter(xs,ys)
# ax.grid()
# plt.show()

# ax00,ax01=plot_segments_and_trail(Candidate)
# ax00.set_title('Original')

# ax10,ax11=plot_segments_and_trail(Candidate_without_dupes)
# ax10.set_title('removed dupes')

# plt.show()
print((100*Candidate_without_dupes.shape[0]/Candidate.shape[0]),len(np.unique(Candidate_without_dupes['segments'])))