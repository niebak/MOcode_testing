import pandas as pd
import numpy as np
from code_functions import *
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(target, candidate):
    target_vector = target.values.flatten()
    candidate_vector = candidate.values.flatten()
    cosine_sim = cosine_similarity([target_vector], [candidate_vector])[0][0]
    return cosine_sim
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
# DB = pd.read_excel('LUW_example.xlsx')
Target = DF_to_segmented_DF(pd.read_excel('data/221005_eksempelsegment001.xlsx'))
Target_features = segments_to_feature_df_with_rev(Target)
Target_features['seg_distance'] = Target_features['seg_distance']/100

Candidate= DF_to_segmented_DF(pd.read_parquet('data/2022-06-05-12-12-09 (1).parquet'))
Candidate_features = segments_to_feature_df_with_rev(Candidate)
Candidate_features['seg_distance'] = Candidate_features['seg_distance']/100

testing = sliding_cosine_similarity(Target_features,Candidate_features)
print(testing)
print(max(testing),np.argmax(testing))