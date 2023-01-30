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


database = 'LUW_example.xlsx'
target_name = 'data/221005_eksempelsegment001.xlsx'


DB = pd.read_excel(database)
Target = DF_to_segmented_DF(pd.read_excel(target_name))

Target_features = (segments_to_feature_df_with_rev(Target))
classify_feature_df(Target_features)
Target_features['seg_distance']=Target_features['seg_distance']*0
num_of_trails = np.unique(DB['name'])[-1]+1
costs=[0]*num_of_trails
paths=[0]*num_of_trails
for i in range(num_of_trails):
    Candidate = DB.loc[DB['name']==i].copy(deep=True)
    Candidate.reset_index(drop=True,inplace=True)
    Candidate_features = (segments_to_feature_df_with_rev(Candidate))
    Candidate_features['seg_distance']=Candidate_features['seg_distance']*0
    classify_feature_df(Candidate_features)
    start_time=time.time()
    cost,path = ViterbiAlgorithm(Candidate_features,Target_features,'curvature','seg_distance','climb','revcurvature','seg_distance','revclimb')
    stop_time = time.time()
    print('\n')
    print(i,cost,path)
    print(f'\nSpent {stop_time-start_time}s finding path!\n')
    costs[i]=cost
    paths[i]=path
print('\nDone!\n')
print(costs)
costs.pop(np.argmin(costs))
best_candidate = np.argmin(costs)
print(f'best trail is {best_candidate}, and the path through it is {paths[best_candidate]}.')
features=['segments','curvature','seg_distance','climb']
winner_segments=np.unique(paths[best_candidate])
winner_raw = DB.loc[DB['name']==best_candidate].copy(deep=True)
winner = winner_raw[winner_raw['segments'].isin(winner_segments)]
winner.reset_index(drop=True,inplace=True)

winner = DF_to_segmented_DF(winner)
winner_feature=segments_to_feature_df(winner) 
# The problem is that the segments doesnt start from zero, they start at a non-zero number and ive based all code on a start from zero system.
classify_feature_df(winner_feature)
print(paths[best_candidate])
print(winner_feature[features])
print(Target_features[features])

ax00,ax01=plot_segments_and_trail(Target)
ax01.legend()
ax10,ax11=plot_segments_and_trail(winner)
ax11.legend()
plt.show()