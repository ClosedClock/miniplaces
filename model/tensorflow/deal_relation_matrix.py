import numpy as np
from get_categories_relation import plot_heatmap

relation = np.load('relation.npz')['arr_0']
N = relation.shape[0]
for i in range(N):
    relation[i][i] = 0

for i in range(N):
    mean = np.mean(relation[:, i])
    relation[:, i] = relation[:, i] - mean

for i in range(N):
    relation[i][i] = 1

plot_heatmap(relation)
np.savez('biased_relation.npz', relation)