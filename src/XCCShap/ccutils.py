import pandas as pd
import numpy as np
from cc.taucc import CoClust
from sklearn.utils import check_array

class XCoClust(CoClust):

    def __init__(self, X, row_labels, col_labels, n_iterations=500, n_iter_per_mode=100, initialization='random', k=30, l=30, row_clusters=None, col_clusters=None, initial_prototypes=None, verbose=False):
        super().__init__(n_iterations, n_iter_per_mode, initialization, k, l, row_clusters, col_clusters, initial_prototypes, verbose)
        self._init_all(X)
        self._row_assignment=row_labels
        self._col_assignment=col_labels
        self._n_row_clusters=len(np.unique(self._row_assignment))
        self._n_col_clusters=len(np.unique(self._col_assignment))


    def assign_samples(self, V):
        dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])
        dataset = dataset/self._tot        
        dataset = self._update_V(dataset)
        V_labels = np.zeros((np.shape(V)[0],1), dtype=float)
        _ , T = self._init_contingency_matrix(0)
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        B = np.nan_to_num(B)
        for i in range(np.shape(V)[0]):
            all_tau = np.sum(dataset[i]*B, axis = 1)
            max_tau = np.max(all_tau)
            equal_solutions = np.where(max_tau == all_tau)[0]
            V_labels[i] = equal_solutions[0]            
        return np.copy(V_labels).tolist()

    def _update_V(self, dataset):
        new_t = np.zeros((np.shape(dataset)[0], self._n_col_clusters), dtype = float)

        for i in range(self._n_col_clusters):
            new_t[:,i] = np.sum(dataset[:,self._col_assignment == i], axis = 1)

        return new_t