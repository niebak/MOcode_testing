import pandas as pd
from code_functions import plot_segments_and_trail,print_df,classify_feature_df
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
import seaborn as sns
import networkx as nx

Feat_DB = pd.read_excel('LUWfeature_example.xlsx')
All_candidates = np.unique(Feat_DB['name'])
# print(All_candidates)
# print(Feat_DB.columns)

Single_candidate = Feat_DB.loc[Feat_DB['name']==1].copy(deep=True)
Single_candidate.reset_index(inplace=True,drop=True)
Single_candidate['state'] = Single_candidate.apply(lambda x: (x['curvature'], x['climb'], x['seg_distance']), axis=1)
classify_feature_df(Single_candidate)
Single_candidate['state_name'] = Single_candidate['class']
print_df(Single_candidate,length=Single_candidate.shape[0])

