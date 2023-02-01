from code_functions import *
from tabulate import tabulate
# from search import *
from VA_improved import *
import time
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(target, candidate):
    target_vector = target.values.flatten()
    candidate_vector = candidate.values.flatten()
    cosine_sim = cosine_similarity([target_vector], [candidate_vector])[0][0]
    return cosine_sim
def score_candidate(df1, df2):
    df1_matrix = df1.values
    df2_matrix = df2.values
    print(df1)
    cos_sim = cosine_similarity(df1_matrix, df2_matrix)[0][0]
    cos_sim_percentage = (cos_sim) / 1 * 100
    return cos_sim_percentage
def sliding_cosine_similarity(target,candidate,features=['curvature','climb','seg_distance']):
    '''
    Uses a sliding window to find the difference between the dataframes.
    Returns a list of the difference between the dataframes, if the window starts at the index.
    '''
    window_size = target.shape[0]
    # print('window size',window_size)
    difference = [np.inf] * (candidate.shape[0] - window_size + 1)
    target_values=target[features]
    for i in range(0,len(difference)):
        subset_candidate = candidate[features].iloc[i:i+window_size]
        difference[i] = calculate_cosine_similarity(target_values,subset_candidate)
    return difference
def print_df(df,headers='keys',tablefmt='github'):
    if df.shape[0]<11:
        length = df.shape[0]
    else:
        length=10
    print('\n')
    print(tabulate(df.iloc[0:length], headers=headers, tablefmt=tablefmt))
    print('\n')
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

rawdatabase = 'LUW_example.xlsx'
database = 'LUWfeature_example.xlsx'
target_name = 'data/221005_eksempelsegment001.xlsx'


DB = pd.read_excel(database)
rawDB = pd.read_excel(rawdatabase)
stats = DB.describe()

Target = DF_to_segmented_DF(pd.read_excel(target_name))
# Target = Target[Target['segments'].isin([0,1,2,3,4,5,6,7,8,9])]
Target_features = (segments_to_feature_df_with_rev(Target))
classify_feature_df(Target_features)
Target_features['seg_distance']=Target_features['seg_distance']/stats['seg_distance'].loc['std']
Target_features['curvature']=Target_features['curvature']/stats['curvature'].loc['std']
Target_features['climb']=Target_features['climb']/stats['climb'].loc['std']

num_of_trails = np.unique(DB['name'])[-1]+2
print(num_of_trails)
costs=[0]*num_of_trails
paths=[0]*num_of_trails
sliw_window_costs=[0]*num_of_trails
sliw_window_paths=[0]*num_of_trails

k=Target.shape[0]
ax0s=[]
ax1s=[]
VA_scores=[]
SW_scores=[]
features = ['curvature','seg_distance','climb']

for i in range(num_of_trails):
    # T0,T1=plot_segments_and_trail(Target)
    # Read the Candidate from the database
    Candidate =  rawDB.loc[rawDB['name']==i].copy(deep=True)
    Candidate_features = (segments_to_feature_df_with_rev(Candidate))
    # Normalising the features with the std from the database. the target is augmented similarly
    Candidate_features['seg_distance']=Candidate_features['seg_distance']/stats['seg_distance'].loc['std']
    Candidate_features['curvature']=Candidate_features['curvature']/stats['curvature'].loc['std']
    Candidate_features['climb']=Candidate_features['climb']/stats['climb'].loc['std']
    classify_feature_df(Candidate_features)
    print(f'\nCandidate {i}')
    # print(Target_features[features])
    # Finding the best part by using the VA 
    start_time=time.time()
    cost,path = ViterbiAlgorithm(Candidate_features,Target_features)
    # cost,path = ViterbiAlgorithm(Candidate_features,Target_features)
    stop_time = time.time()
    # looking at the winner-sequence
    VA_winner_path = Candidate_features.loc[Candidate_features['segments'].isin(path)].copy(deep=True)
    VA_winner_path.reset_index(drop=True,inplace=True)

    VA_winner = Candidate.loc[Candidate['segments'].isin(path)].copy(deep=True)
    VA_winner.reset_index(drop=True,inplace=True)
    VA_score = score_candidate(Target_features[features],VA_winner_path[features])
    print(f'VA spent {round(stop_time-start_time,3)}s, score of {round(VA_score,2)}%, or cost of {cost}')
    print(path)
    # # print(VA_winner_path[features])
    VA_scores.append(VA_score)
    # print_df(VA_winner_path)
    # VA0,VA1=plot_segments_and_trail(VA_winner)
    # Finding the best path by using a sliding window technuqie
    start_time=time.time()
    difference = sliding_window_features(Target_features,Candidate_features)
    best_start = np.argmin(difference)
    best_end = best_start + Target_features.shape[0]
    best_segments = np.linspace(best_start,best_end-1,best_end-best_start)
    stop_time = time.time()

    SW_winner_path = Candidate_features.loc[Candidate_features['segments'].isin(best_segments)].copy(deep=True)
    SW_winner_path.reset_index(drop = True,inplace=True)
    SW_winner = Candidate.loc[Candidate['segments'].isin(best_segments)].copy(deep=True)
    SW_winner.reset_index(drop = True,inplace=True)
    SW_score = score_candidate(Target_features[features],SW_winner_path[features])
    print(f'\nSW spent {round(stop_time-start_time,3)}s, score of {round(SW_score,2)}%')
    print(best_segments)
    # print(SW_winner_path[features])
    SW_scores.append(SW_score)





    # Find some method to create a better VA path that includes the correct segments and then see 
    # The differences between the two methods using the copare dataframes function






    # VA_winner_path = pd.DataFrame()
    # for i in range(0,len(path)):
    #     new = Candidate_features.iloc[i]
    #     print(new)
    #     VA_winner_path = pd.concat([VA_winner_path,new],ignore_index=True)
    # print(VA_winner_path)
    # VAdiffs = compare_dataframes(VA_winner_path[features],Target_features[features])
    # SWdiffs = compare_dataframes(SW_winner_path[features],Target_features[features])
    
    print(VAdiffs)
    print('\n')
    print(SWdiffs)
    
    costs[i]=cost
    paths[i]=path
    sliw_window_costs[i]=difference[best_start]
    sliw_window_paths[i]=best_segments
    
print('\nDone!\n')
print(f'VA {np.mean(VA_scores)}\n')
print(f'SW {np.mean(SW_scores)}\n')
costs.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
costs.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
sliw_window_costs.pop(np.argmin(sliw_window_costs)) # This has to be removed, as the target is present in the database
sliw_window_costs.pop(np.argmin(sliw_window_costs)) # This has to be removed, as the target is present in the database
best_candidate = np.argmin(costs)
print(f'best trail is {best_candidate}, and the path through it is {paths[best_candidate]}, with a cost of {costs[best_candidate]}')
features=['segments','curvature','seg_distance','climb']


sw_winner=np.argmin(sliw_window_costs)
sw_segments = sliw_window_paths[sw_winner]
print(f'According to sliding window it is {sw_winner}, and these segments: {sw_segments}.\n')

complete_winner = rawDB.loc[rawDB['name']==best_candidate].copy(deep=True)
# print(complete_winner)
winner_path = complete_winner.loc[complete_winner['segments'].isin(paths[best_candidate])].copy(deep=True)
winner_path.reset_index(drop=True,inplace=True)

sw_complete_winner = rawDB.loc[rawDB['name']==sw_winner].copy(deep=True)
# print(complete_winner)
sw_winner_path = sw_complete_winner.loc[sw_complete_winner['segments'].isin(sw_segments)].copy(deep=True)
sw_winner_path.reset_index(drop=True,inplace=True)


Target_features = (segments_to_feature_df_with_rev(Target))


# # print(winner_path)

# ax00,ax01=plot_segments_and_trail(Target)
# ax01.legend()
# ax10,ax11=plot_segments_and_trail(winner_path)
# ax11.legend()
# ax20,ax21=plot_segments_and_trail(sw_winner_path)
# ax21.legend()
# # plt.show()

# print(tabulate(segments_to_feature_df_with_rev(Target),headers='keys',tablefmt='github'))
# print(tabulate(segments_to_feature_df_with_rev(winner_path),headers='keys',tablefmt='github'))
# print(tabulate(segments_to_feature_df_with_rev(sw_winner_path),headers='keys',tablefmt='github'))