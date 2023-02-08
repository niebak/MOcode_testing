from VA_improved import ViterbiAlgorithm,Differential_ViterbiAlgorithm,tanh,sliding_window_features
from code_functions import *
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

rawdatabase = 'LUW_example.xlsx'
database = 'LUWfeature_example.xlsx'
target_name = 'data/221005_eksempelsegment001.xlsx'

DB = pd.read_excel(database)
rawDB = pd.read_excel(rawdatabase)
stats = DB.describe()

Target = DF_to_segmented_DF(pd.read_excel(target_name))
Target_features = (segments_to_feature_df_with_rev(Target))
classify_feature_df(Target_features)

Target_features['seg_distance']=Target_features['seg_distance']/stats['seg_distance'].loc['std']
Target_features['curvature']=Target_features['curvature']/stats['curvature'].loc['std']
Target_features['climb']=Target_features['climb']/stats['climb'].loc['std']

num_of_trails = np.unique(DB['name'])[-1]+2
costs=[0]*num_of_trails
paths=[0]*num_of_trails
diffcosts=[0]*num_of_trails
diffpaths=[0]*num_of_trails
test = 'Valtitude' in Target.columns
print(f'\nTarget: {test}')
grouped_target = Target.groupby('segments')
Target_diff = grouped_target.mean().reset_index()


for i in range(num_of_trails):
    Candidate =  (rawDB.loc[rawDB['name']==i].copy(deep=True))
    Candidate_vector = coordinate_to_vector_dataframe(Candidate)
    
    Candidate['Vposition_long']=Candidate_vector['Vposition_long']
    Candidate['Vposition_lat']=Candidate_vector['Vposition_lat']
    Candidate['Valtitude']=Candidate_vector['Valtitude']

    grouped_Candidate = Candidate.groupby('segments')
    Candidate_diff = grouped_Candidate.mean().reset_index()


    Candidate_features = (segments_to_feature_df_with_rev(Candidate))
    Candidate_features['seg_distance']=Candidate_features['seg_distance']/stats['seg_distance'].loc['std']
    Candidate_features['curvature']=Candidate_features['curvature']/stats['curvature'].loc['std']
    Candidate_features['climb']=Candidate_features['climb']/stats['climb'].loc['std']

    classify_feature_df(Candidate_features)
    start_time=time.time()
    cost,path = ViterbiAlgorithm(Candidate_features,Target_features)
    stop_time = time.time()
    print(i,stop_time-start_time)
    test='Valtitude' in Candidate.columns
    print(f'Candidate: {test}')

    start_time=time.time()
    diffcost,diffpath = Differential_ViterbiAlgorithm(Candidate_diff,Target_diff)
    stop_time = time.time()
    print(i,stop_time-start_time)
    costs[i]=cost
    paths[i]=path

    diffcosts[i]=cost
    diffpaths[i]=path
    print('\n')
print('\nDone!\n')
costs.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
costs.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
diffcosts.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
diffcosts.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
best_candidate = np.argmin(costs)
diffbest_candidate = np.argmin(diffcosts)
print(f'best trail is {best_candidate}, and the path through it is {paths[best_candidate]}, with a cost of {costs[best_candidate]}')
print(f'best diff trail is {diffbest_candidate}, and the path through it is {diffpaths[diffbest_candidate]}, with a cost of {diffcosts[diffbest_candidate]}')

complete_winner = rawDB.loc[rawDB['name']==best_candidate].copy(deep=True)
# print(complete_winner)
winner_path = complete_winner.loc[complete_winner['segments'].isin(paths[best_candidate])].copy(deep=True)
winner_path.reset_index(drop=True,inplace=True)

diffcomplete_winner = rawDB.loc[rawDB['name']==diffbest_candidate].copy(deep=True)
# print(complete_winner)
diffwinner_path = diffcomplete_winner.loc[diffcomplete_winner['segments'].isin(diffpaths[diffbest_candidate])].copy(deep=True)
diffwinner_path.reset_index(drop=True,inplace=True)

ax00,ax01 = plot_segments_and_trail(winner_path)
ax10,ax11 = plot_segments_and_trail(diffwinner_path)

plt.show()