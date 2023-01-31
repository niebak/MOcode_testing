from code_functions import *
from search import *
from tabulate import tabulate
import pandas as pd
import time
def compare_dataframes(df1, df2):
    # Ensure that df1 is the shorter dataframe
    if df1.shape[0] > df2.shape[0]:
        df1, df2 = df2, df1
    # Create an empty DataFrame to store the comparison results
    window_size= df1.shape[0]
    results = pd.DataFrame(columns=['start_index_df1', 'end_index_df1', 'start_index_df2', 'end_index_df2', 'difference'])
    # Loop through the rows of df1
    for i in range(df1.shape[0] - window_size + 1):
        start_index_df1 = i
        end_index_df1 = i + window_size
        # Loop through the rows of df2
        for j in range(df2.shape[0] - window_size + 1):
            start_index_df2 = j
            end_index_df2 = j + window_size
            # Calculate the difference between the corresponding window in df1 and df2
            difference = (df1.iloc[start_index_df1:end_index_df1] - df2.iloc[start_index_df2:end_index_df2]).abs().sum().sum()
            results = results.append({'start_index_df1': start_index_df1, 'end_index_df1': end_index_df1, 'start_index_df2': start_index_df2, 'end_index_df2': end_index_df2, 'difference': difference}, ignore_index=True)
    return results
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


# DB = pd.read_excel('LUW_example.xlsx')
Target = DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))
Target_features = segments_to_feature_df_with_rev(Target)
Target_features['seg_distance'] = Target_features['seg_distance']/100

Candidate= DF_to_segmented_DF(pd.read_parquet('data/2022-06-05-12-12-09 (1).parquet'))
Candidate_features = segments_to_feature_df_with_rev(Candidate)
Candidate_features['seg_distance'] = Candidate_features['seg_distance']/100

start_time = time.time()
testing = sliding_window_features(Target_features,Candidate_features)
stop_time = time.time()

cost,path = ViterbiAlgorithm(Candidate_features,Target_features,'curvature','seg_distance','climb','revcurvature','seg_distance','revclimb')

# print(stop_time-start_time)
# # print(testing)
# print(np.min(testing),np.argmin(testing))

first_segment = np.argmin(testing)
last_segment = first_segment + Target_features.shape[0]
best_segments=np.linspace(first_segment,last_segment,last_segment-first_segment+1)

print(path)


winner = Candidate.loc[Candidate['segments'].isin(best_segments)].copy(deep=True)
winner.reset_index(drop=True,inplace=True)

VA = Candidate.loc[Candidate['segments'].isin(path)].copy(deep=True)
VA.reset_index(drop=True,inplace=True)

ax00,ax01 = plot_segments_and_trail(Target)
ax10,ax11 = plot_segments_and_trail(winner)
ax11.legend()
ax20,ax21 = plot_segments_and_trail(VA)
ax21.legend()

plt.show()