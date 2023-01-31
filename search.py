import numpy as np
def ViterbiAlgorithm(C1,C0,feature1,feature2,feature3,revfeature1,revfeature2,revfeature3):
    '''C1 is the candidate, C0 is the target. Return Cost,Path'''
    # Initialise matrixes and renaming
    AcC=np.ones([C1.shape[0],C0.shape[0]])*np.nan # Accumulated cost
    Pointer=np.ones([C1.shape[0],C0.shape[0]])*np.nan # Pointers
    source=C1
    target=C0
    # Some startup definitions
    Target=C0.iloc[1]
    Trans01,Route=Find_trans_matrix(C1,Target,feature1,feature2,feature3,revfeature1,revfeature2,revfeature3)
    # Initial position
    for s in range(len(source)):
        AcC[s,0]=np.min(Trans01[s])
        Pointer[s,0]=np.argmin(Trans01[s])
    # The rest of the posititons
    for o in range(1,len(target)): # Length of observations/target
        Target=C0.iloc[o]
        Trans01,Route=Find_trans_matrix(C1,C0.iloc[o],feature1,feature2,feature3,revfeature1,revfeature2,revfeature3) # Finding the trans. matrix
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
def Find_trans_matrix(C1,CompCourse,feature1,feature2,feature3,revfeature1,revfeature2,revfeature3):
    Trans=np.ones([C1.shape[0],C1.shape[0]])*np.inf
    Routes=np.ones([C1.shape[0],C1.shape[0]])*np.inf
    # creating the Transition matrix
    for i in range(0,Trans.shape[1]):
        for j in range(0,Trans.shape[0]):
            if((i+1)==(j)):
                Trans[i,j]=(Dist(C1,CompCourse,j,f1=feature1,f2=feature2,f3=feature3))
                Routes[i,j]=i*10+j
            if((j+1)==(i)):
#                 if(j<=4):
                Trans[i,j]=(Dist(C1,CompCourse,j,f1=revfeature1,f2=revfeature2,f3=revfeature3))
                Routes[i,j]=i*10+j
#         print(Trans)
    return Trans,Routes
def Dist(course1,course2,where,f1='f1',f2='f2',f3='f3',rf1='rf1',rf2='rf2',rf3='rf3',w=[1,1,1]):
    c1f1=course1[f1].iloc[where]
    c1f2=course1[f2].iloc[where]
    c1f3=course1[f3].iloc[where]
    
    c2f1=course2[f1]
    c2f2=course2[f2]
    c2f3=course2[f3]
    
    d1=(c2f1-c1f1)**2
    d2=(c2f2-c1f2)**2
    d3=(c2f3-c1f3)**2
    
    d=np.sqrt(abs(w[0]*d1+w[1]*d2+w[2]*d3))
    return d
def tanh(x,k=1):
    upper = np.exp(x*k)-np.exp(-x*k)
    lower = np.exp(x*k)+np.exp(-x*k)
    return upper/lower