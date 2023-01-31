from code_functions import *
from tabulate import tabulate
from search import *
import time
def print_df(df,headers='keys',tablefmt='github'):
    if df.shape[0]<11:
        length = df.shape[0]
    else:
        length=10
    print('\n')
    print(tabulate(df.iloc[0:length], headers=headers, tablefmt=tablefmt))
    print('\n')


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

k=Target.shape[0]
ax0s=[]
ax1s=[]
for i in range(num_of_trails):
    Candidate =  rawDB.loc[rawDB['name']==i].copy(deep=True)
    # Candidate_features = DB.loc[DB['name']==i].copy(deep=True)
    # Candidate.reset_index(drop=True,inplace=True)
    Candidate_features = (segments_to_feature_df_with_rev(Candidate))
    Candidate_features['seg_distance']=Candidate_features['seg_distance']/stats['seg_distance'].loc['std']
    Candidate_features['curvature']=Candidate_features['curvature']/stats['curvature'].loc['std']
    Candidate_features['climb']=Candidate_features['climb']/stats['climb'].loc['std']
    classify_feature_df(Candidate_features)
    start_time=time.time()
    cost,path = ViterbiAlgorithm(Candidate_features,Target_features,'curvature','seg_distance','climb','revcurvature','seg_distance','revclimb')
    stop_time = time.time()
    # print('\n')
    # print(i,cost,path)
    print(f'\nSpent {stop_time-start_time}s finding path with cost {cost}!')
    score = 100*(1-tanh(cost, k=1/(0.09*k)))
    print(f'This is a match of {round(score,2)}% for trail {i}\n')

    # ax0,ax1=plot_segments_and_trail(Candidate)
    # ax0.set_xlabel(i)
    # ax0s.append(ax0)
    # ax1s.append(ax1)

    costs[i]=cost
    paths[i]=path
print('\nDone!\n')
# print(costs)
costs.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
# costs.pop(np.argmin(costs)) # This has to be removed, as the target is present in the database
best_candidate = np.argmin(costs)
print(f'best trail is {best_candidate}, and the path through it is {paths[best_candidate]}, with a cost of {costs[best_candidate]}')
features=['segments','curvature','seg_distance','climb']

winner_segments=np.unique(paths[best_candidate])
winner_raw = rawDB.loc[rawDB['name']==best_candidate].copy(deep=True)
winner = winner_raw[winner_raw['segments'].isin(winner_segments)]
winner.reset_index(drop=True,inplace=True)
# print(winner)
winner_features = segments_to_feature_df_with_rev(winner)

complete_winner = rawDB.loc[rawDB['name']==best_candidate].copy(deep=True)
# print(complete_winner)
winner_path = complete_winner.loc[complete_winner['segments'].isin(paths[best_candidate])].copy(deep=True)
winner_path.reset_index(drop=True,inplace=True)

winner = DF_to_segmented_DF(winner)
winner_feature=segments_to_feature_df(winner) 
classify_feature_df(winner_feature)

Target_features = (segments_to_feature_df_with_rev(Target))
print(paths[best_candidate])
print(winner_features[features])
print(Target_features[features])

# print(winner_path)

ax00,ax01=plot_segments_and_trail(Target)
ax01.legend()
ax10,ax11=plot_segments_and_trail(winner_path)
ax11.legend()
plt.show()