import numpy as np
from scipy.spatial import distance
import pandas as pd
def ViterbiAlgorithm(C1,C0):
    '''C1 is the candidate, C0 is the target. Return Cost,Path'''
    # Initialise matrixes and renaming
    AcC=np.ones([C1.shape[0],C0.shape[0]])*np.nan # Accumulated cost
    Pointer=np.ones([C1.shape[0],C0.shape[0]])*np.nan # Pointers
    source=C1
    target=C0
    # Some startup definitions
    Target=C0.iloc[0]
    Trans01,Route=Find_trans_matrix(C1,Target)
    # Initial position
    for s in range(len(source)):
        AcC[s,0]=np.min(Trans01[s])
        Pointer[s,0]=np.argmin(Trans01[s])
    # The rest of the posititons
    for o in range(1,len(target)): # Length of observations/target
        Target=C0.iloc[o]
        Trans01,Route=Find_trans_matrix(C1,C0.iloc[o]) # Finding the trans. matrix
        for s in range(0,len(source)): # Setting the best value possible
            Routes=np.where(Trans01[s]<np.inf)[0]

            if(len(Routes)>=2): # If for some reason we have more than two I'll need to add some code here
                path1=Routes[0]
                path2=Routes[1]
                AcC[s,o]=min(AcC[path1,o-1]+Trans01[path1,s],AcC[path2,o-1]+Trans01[path2,s])
                if (AcC[s,o]==AcC[path1,o-1]+Trans01[path1,s]):
                    bestpath=path1
                else:
                    bestpath=path2
                Pointer[s,o]=bestpath
            else:
                path1=Routes[0] # If only a single path is possible
                AcC[s,o]=AcC[path1,o-1]+Trans01[path1,s]
                Pointer[s,o]=path1
    bestend=np.argmin(AcC[:,len(target)-1])
    k=bestend
    bestpath=[]
    for o in range(len(target)-1,-1,-1):
        bestpath.insert(0,k)
        k=int(Pointer[int(k),int(o)])
    cost=min(AcC[:,len(target)-2])
             
    return cost,bestpath
def Find_trans_matrix(C1,TargetRow):
    Trans=np.ones([C1.shape[0],C1.shape[0]])*np.inf
    Routes=np.ones([C1.shape[0],C1.shape[0]])*np.inf
    # creating the Transition matrix
    for i in range(0,Trans.shape[1]):
        for j in range(0,Trans.shape[0]):
            if((i+1)==(j)):
                Trans[i,j]=(Dist(C1,TargetRow,j))
                Routes[i,j]=i*10+j
            if((j+1)==(i)):
#                 if(j<=4):
                Trans[i,j]=(Dist(C1,TargetRow,j,features = ['revcurvature','revclimb','seg_distance']))
                Routes[i,j]=i*10+j
#         print(Trans)
    return Trans,Routes
def Dist(course1,course2,where,features = ['curvature','climb','seg_distance']):
    df1 = course1[features].iloc[where].values
    df2 = course2[features].values

    np.nan_to_num(df1,copy=False)
    np.nan_to_num(df2,copy=False)

    d = distance.euclidean(df1,df2) 
    return d
def tanh(x,k=1):
    upper = np.exp(x*k)-np.exp(-x*k)
    lower = np.exp(x*k)+np.exp(-x*k)
    return upper/lower
def compare_dataframes(df1, df2):
    if df1.shape != df2.shape:
        return 0

    differences = []
    for i in range(df1.shape[0]):
        row1 = df1.iloc[i, :].values
        row2 = df2.iloc[i, :].values
        differences.append(np.linalg.norm(row1 - row2))
    return differences
def Differential_ViterbiAlgorithm(C1,C0):
    '''C1 is the candidate, C0 is the target. Return Cost,Path'''
    # Initialise matrixes and renaming
    AcC=np.ones([C1.shape[0],C0.shape[0]])*np.nan # Accumulated cost
    Pointer=np.ones([C1.shape[0],C0.shape[0]])*np.nan # Pointers
    source=C1
    target=C0
    # Some startup definitions
    Target=C0.iloc[0]
    Trans01,Route=Diff_Find_trans_matrix(C1,Target)
    # Initial position
    for s in range(len(source)):
        AcC[s,0]=np.min(Trans01[s])
        Pointer[s,0]=np.argmin(Trans01[s])
    # The rest of the posititons
    for o in range(1,len(target)): # Length of observations/target
        Target=C0.iloc[o]
        Trans01,Route=Diff_Find_trans_matrix(C1,C0.iloc[o]) # Finding the trans. matrix
        for s in range(0,len(source)): # Setting the best value possible
            Routes=np.where(Trans01[s]<np.inf)[0]

            if(len(Routes)>=2): # If for some reason we have more than two I'll need to add some code here
                path1=Routes[0]
                path2=Routes[1]
                AcC[s,o]=min(AcC[path1,o-1]+Trans01[path1,s],AcC[path2,o-1]+Trans01[path2,s])
                if (AcC[s,o]==AcC[path1,o-1]+Trans01[path1,s]):
                    bestpath=path1
                else:
                    bestpath=path2
                Pointer[s,o]=bestpath
            else:
                path1=Routes[0] # If only a single path is possible
                AcC[s,o]=AcC[path1,o-1]+Trans01[path1,s]
                Pointer[s,o]=path1
    bestend=np.argmin(AcC[:,len(target)-1])
    k=bestend
    bestpath=[]
    for o in range(len(target)-1,-1,-1):
        bestpath.insert(0,k)
        k=int(Pointer[int(k),int(o)])
    cost=min(AcC[:,len(target)-2])
             
    return cost,bestpath
def Diff_Find_trans_matrix(C1,TargetRow):
    Trans=np.ones([C1.shape[0],C1.shape[0]])*np.inf
    Routes=np.ones([C1.shape[0],C1.shape[0]])*np.inf
    # creating the Transition matrix
    for i in range(0,Trans.shape[1]):
        for j in range(0,Trans.shape[0]):
            if((i+1)==(j)):
                Trans[i,j]=(Dist(C1,TargetRow,j,features=['Vposition_lat','Vposition_long','Valtitude']))
                Routes[i,j]=i*10+j
            if((j+1)==(i)):
#                 if(j<=4):
                Trans[i,j]=(revDist(C1,TargetRow,j,features = ['Vposition_lat','Vposition_long','Valtitude']))
                Routes[i,j]=i*10+j
#         print(Trans)
    return Trans,Routes
def revDist(course1,course2,where,features = ['curvature','climb','seg_distance']):
    df1 = course1[features].iloc[where].values
    df2 = course2[features].values
    df1 = np.multiply(df1,-1)

    np.nan_to_num(df1,copy=False)
    np.nan_to_num(df2,copy=False)

    d = distance.euclidean(df1,df2) 
    return d
def sliding_window_features(target,candidate,features=['curvature','climb','seg_distance']):
    '''
    Uses a sliding window to find the difference between the dataframes.
    Returns a list of the difference between the dataframes, if the window starts at the index.
    '''
    window_size = target.shape[0]
    # print('window size',window_size)
    difference = [np.inf] * (candidate.shape[0] - window_size + 1)
    target_values=target[features].to_numpy()
    for i in range(0,len(difference)):
        subset_candidate = candidate[features].iloc[i:i+window_size].to_numpy()
        difference[i] = np.linalg.norm(target_values - subset_candidate)
    return difference