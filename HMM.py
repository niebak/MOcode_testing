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

transition_counts = Single_candidate.groupby(['state_name', 'state']).size().reset_index(name='count')

grouped = Single_candidate.groupby('state_name')

transitions = {}

for name, group in grouped:
    total_count = len(group)

    transition_matrix = np.zeros((len(group), len(group)))

    for i, row in group.iterrows():
        j = group.index.get_loc(i)
        transition_matrix[j, :] = transition_counts.loc[
            (transition_counts['state_name'] == name) & (transition_counts['state'] == row['state']),
            'count'].values / total_count

    transitions[name] = transition_matrix

n_components = len(transitions)
model = hmm.MultinomialHMM(n_components=n_components)

A = np.array(list(transitions.values()))
model.transmat_ = A

X = Single_candidate['state_name'].apply(lambda x: list(transitions.keys()).index(x)).values.reshape(-1, 1)
model.fit(X)

trans_mat = model.transmat_

labels = dict(enumerate(np.unique(Single_candidate['state_name'].to_numpy())))

sns.heatmap(trans_mat, cmap="Blues", annot=True, square=True, cbar=False, xticklabels=labels.values(), yticklabels=labels.values())
plt.xlabel("To state")
plt.ylabel("From state")
plt.show()

G = nx.DiGraph(trans_mat)

pos = nx.circular_layout(G)

# Adjust node positions
theta = np.linspace(0, 2*np.pi, n_components, endpoint=False)
radius = 1.2
for i, node in enumerate(G.nodes):
    pos[node] = np.array([radius*np.cos(theta[i]), radius*np.sin(theta[i])])

nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, font_size=10, arrowsize=20, arrowstyle='->', width=2)

# Add edge labels
edge_labels = {}
for u, v, d in G.edges(data=True):
    edge_labels[(u, v)] = f'{d["weight"]:.2f}'

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.show()
