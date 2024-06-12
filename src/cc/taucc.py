import logging
from random import choice
from random import randint
from time import time
from typing import List

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import adjusted_rand_score as ari


class CoClust(BaseEstimator, ClusterMixin, TransformerMixin):
    """ CoStar algorithm (Ienco, 2013).

    CoStar is an algorithm created to deal with multi-view data.
    It finds automatically the best number of row / column clusters.

    Parameters
    ------------

    n_iterations : int, optional, default: 300
        The number of iterations to be performed.

    min_n_row_clusters : int, optional, default: 2,
        The number of clusters for rows, should be at least 2

    init : {'standard_with_merge', 'standard_without_merge', 'random', 'kmeans'}, optional, default: 'standard_without_merge'
        The initialization methods:

        'standard_with_merge' : the initial clusters are discrete (one item per cluster), but identical
            rows are merged into one single cluster

        'standard_without_merge' : the initial clusters are discrete (one item per cluster)

        'random' : the initial clusters are random and are less than the number of elements

        'kmeans' : the initial cluster for both rows and columns are chosen by an execution of kmeans.
                    the number of cluster is equal to number_of_elements * scaling_factor

    scaling_factor : float, optional, default: 0,01
        The scaling factor to use to define the number of cluster desired in case of 'kmeans' initialization method.

    Attributes
    -----------

    rows_ : array, length n_rows
        Results of the clustering on rows. `rows[i]` is `c` if
        row `i` is assigned to cluster `c`. Available only after calling ``fit``.

    columns_ : array, length n_columns
        Results of the clustering on columns. `columns[i]` is `c` if
        column `i` is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    References
    ----------

    * Ienco, Dino, et al., 2013. `Parameter-less co-clustering for star-structured heterogeneous data.`
        Data Mining and Knowledge Discovery 26.2: 217-254

    """

    def __init__(self, n_iterations=500, n_iter_per_mode = 100, initialization= 'random', k = 30, l = 30, row_clusters = np.zeros(1), col_clusters = np.zeros(1), initial_prototypes = np.zeros(1), verbose = False):
        """
        Create the model object and initialize the required parameters.

        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type n_iter_per_mode: int
        :param n_iter_per_mode: the max number of sub-iterations for eac iteration and each mode
        :type initialization: string
        :param initialization: the initialization method, one of {'random', 'discrete', 'random_optimal', 'extract_centroids', 'customized'}
        :type k: int
        :param k: number of clusters on rows. It is only needed if initialization in {'random', 'extract_centroids'}. In 'random' k and l can be equal to 0 . 
        :type l: int
        :param l: number of clusters on columns. It is only needed if initialization in {'random', 'extract_centroids'}. In 'random' k and l can be equal to 0 . 
        :type row_clusters: np.array of integers
        :param row_clusters: initial cluster assignments of rows. Only needed if initialization in {}
        :type col_clusters: np.array of integers
        :param col_clusters: initial cluster assignments of columns. Only needed if initialization in {}
        :type initial_prototypes: np.array of integers
        :param initial_prototypes: ???
        :type verbose: boolean
        :param verbose: if True, it prints details of the computations
        """

        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.k = k
        self.l = l
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.initial_prototypes = initial_prototypes
        self.verbose = verbose

        # these fields will be available after calling fit
        self.row_labels_ = None
        self.column_labels_ = None
        self.execution_time_ = None

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None

        self._dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])

        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            
        # the number of documents and the number of features in the data (n_rows and n_columns)
        self._n_documents = self._dataset.shape[0]
        self._n_features = self._dataset.shape[1]

        # the number of row/ column clusters
        self._n_row_clusters = 0
        self._n_col_clusters = 0

        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._row_assignment = np.zeros(self._n_documents)
        self._col_assignment = np.zeros(self._n_features)


        self._tot = np.sum(self._dataset)
        self._dataset = self._dataset/self._tot
        self.tau_x = []
        self.tau_y = []
        
        if (self.initialization == 'discrete') or (self.initialization == 'random_optimal'):
            self._discrete_initialization()
        elif self.initialization == 'random':
            self._random_initialization()
        elif self.initialization == 'extract_centroids':
            self._extract_centroids_initialization()
        else:
            raise ValueError("The only valid initialization methods are: random, discrete, extract_centroids")


        logging.debug("[INFO] Row's initialization: {0}".format(list(self._row_assignment)))
        logging.debug("[INFO] Col's initialization: {0}".format(list(self._col_assignment)))


    def fit(self, V, y=None):
        """
        Fit CoClust to the provided data.

        Parameters
        -----------

        V : array-like or sparse matrix;
            shape of the matrix = (n_documents, n_features)

        y : unused parameter

        Returns
        --------

        self

        """

        # Initialization phase
        self._init_all(V)

        self._T = self._init_contingency_matrix(0)[1]
        tau_x, tau_y = self.compute_taus()
        
        self.tau_x.append(tau_x)
        self.tau_y.append(tau_y)

        start_time = time()

        # Execution phase
        self._actual_n_iterations = 0
        actual_n_iterations = 0 
        
        while actual_n_iterations < self.n_iterations:
            actual_iteration_x = 0    
            cont = True
            while cont:
                # perform a move within the rows partition
                cont = self._perform_row_move()

                actual_iteration_x += 1
                self._actual_n_iterations +=1 
                
                if actual_iteration_x > self.n_iter_per_mode:
                    cont = False
                if self.verbose:
                    self._T = self._init_contingency_matrix(0)[1]
                    tau_x, tau_y = self.compute_taus()
                    
                    self.tau_x.append(tau_x)
                    self.tau_y.append(tau_y)
            
            
            actual_iteration_y = 0
            cont = True
            while cont:
                # perform a move within the rows partition
                cont = self._perform_col_move()

                actual_iteration_y += 1
                self._actual_n_iterations +=1 

                if actual_iteration_y > self.n_iter_per_mode:
                    cont = False
                if self.verbose:
                    self._T = self._init_contingency_matrix(0)[1]
                    tau_x, tau_y = self.compute_taus()
                    
                    self.tau_x.append(tau_x)
                    self.tau_y.append(tau_y)
                
            if (actual_iteration_x == 1) and (actual_iteration_y == 1):
                actual_n_iterations = self.n_iterations
            else:
                actual_n_iterations += 1
            

        end_time = time()
        if not self.verbose:
            self._T = self._init_contingency_matrix(1)[1]
            tau_x, tau_y = self.compute_taus()
            
            self.tau_x.append(tau_x)
            self.tau_y.append(tau_y)

        execution_time = end_time - start_time
        logging.info('#####################################################')
        logging.info("[INFO] Execution time: {0}".format(execution_time))

        # clone cluster assignments and transform in lists
        self.row_labels_ = np.copy(self._row_assignment).tolist()
        self.column_labels_ = np.copy(self._col_assignment).tolist()
        self.execution_time_ = execution_time

        logging.info("[INFO] Number of row clusters found: {0}".format(self._n_row_clusters))
        logging.info("[INFO] Number of column clusters found: {0}".format(self._n_col_clusters))

        return self



    def _discrete_initialization(self):
        
        # simply assign each row to a row cluster and each column of a view to a column cluster
        self._n_row_clusters = self._n_documents
        self._n_col_clusters = self._n_features

        # assign each row to a row cluster
        self._row_assignment = np.arange(self._n_documents)

        # assign each column to a cluster
        self._col_assignment = np.arange(self._n_features)



    def _random_initialization(self):

        if (self.k > self._n_documents) or (self.l > self._n_features):
            raise ValueError("The number of clusters must be <= the number of objects, on both dimensions")
        if self.k == 0 :
            self._n_row_clusters = np.random.choice(self._n_documents)
        else:
            self._n_row_clusters = self.k
        if self.l == 0:
            self._n_col_clusters = np.random.choice(self._n_features)
        else:
            self._n_col_clusters = self.l

        # assign each row to a row cluster
        if self._n_row_clusters < self._n_documents:
            self._row_assignment = np.random.choice(self._n_row_clusters, size = self._n_documents)
        else:
            self._row_assignment = np.arange(self._n_row_clusters).astype(int)
        # assign each column to a cluster
        
        if self._n_col_clusters < self._n_features:
            self._col_assignment = np.random.choice(self._n_col_clusters, size = self._n_features)
        else:
            self._col_assignment = np.arange(self._n_col_clusters).astype(int)
            
        self._check_clustering(0)
        self._check_clustering(1)

        

    def _extract_centroids_initialization(self):
        if (self.k > self._n_documents) or (self.l > self._n_features) or (self.k <= 0) or (self.l <= 0):
            raise ValueError("The number of clusters must be <= the number of objects and greater than 0, on both dimensions")

        self._n_row_clusters = self.k
        self._n_col_clusters = self.l

        if self.k == self._n_documents:
            self._row_assignment = np.arange(self._n_documents)
        if self.l == self._n_features:
            self._col_assignment = np.arange(self._n_features)
        if (self.k < self._n_documents) or (self.l < self._n_features):
        
            a = np.random.choice(self._n_documents, self._n_row_clusters, replace = False)
            T = self._dataset[a]
            S = np.repeat(np.sum(self._dataset, axis = 1).reshape((-1,1)), repeats = self._dataset.shape[1], axis = 1)
            B = np.nan_to_num(self._dataset/np.sum(self._dataset, axis = 0) - S)

            for i in range(self._n_documents):
                all_tau = np.sum(B[i]*T, axis = 1)
                max_tau = np.max(all_tau)
                if max_tau > 0:
                
                    e_max = np.where(max_tau == all_tau)[0][0]
                    self._row_assignment[i] = e_max
                else:
                    self._row_assignment[i] = self._n_row_clusters
                    self._n_row_clusters +=1
            self._check_clustering(0)
            
            
            dataset, T = self._init_contingency_matrix(1)
            
            b = np.random.choice(self._n_features, self._n_col_clusters, replace = False)        
            T = dataset[:,b].T
            dataset = dataset.T
            S = np.repeat(np.sum(dataset, axis = 1).reshape((-1,1)), repeats = dataset.shape[1], axis = 1)
            B = np.nan_to_num(dataset/np.sum(dataset, axis = 0) - S)
            for i in range(self._n_features):
                all_tau = np.sum(B[i]*T, axis = 1)
                max_tau = np.max(all_tau)

                if max_tau > 0:
                
                    e_max = np.where(max_tau == all_tau)[0][0]
                    self._col_assignment[i] = e_max
                else:
                    self._col_assignment[i] = self._n_col_clusters
                    self._n_col_clusters +=1

        
        self._check_clustering(1)
        self._init_k = self._n_row_clusters
        self._init_l = self._n_col_clusters
        


    def _check_clustering(self, dimension):
        if dimension == 1:
            k = len(set(self._col_assignment))
            h = [j for j in range(k) if j not in set(self._col_assignment)]
            p = 0
            for i in set(self._col_assignment):
                if i >=k:
                    self._col_assignment[self._col_assignment==i] = h[p]
                    p +=1
            self._n_col_clusters = k
        elif dimension == 0:
            k = len(set(self._row_assignment))
            h = [j for j in range(k) if j not in set(self._row_assignment)]
            p = 0
            for i in set(self._row_assignment):
                if i >=k:
                    self._row_assignment[self._row_assignment==i] = h[p]
                    p +=1
            self._n_row_clusters = k  
        
            
    def _init_contingency_matrix(self, dimension):

        dataset = self._update_dataset(dimension)

        new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        if dimension == 0:   
            for i in range(self._n_row_clusters):
                new_t[i] = np.sum(dataset[self._row_assignment == i], axis = 0)
        else:
            for i in range(self._n_col_clusters):
                new_t[:,i] = np.sum(dataset[:,self._col_assignment == i], axis = 1)
        
        return dataset, new_t

    def _update_dataset(self, dimension):
        if dimension == 0:
            new_t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)

            for i in range(self._n_col_clusters):
                new_t[:,i] = np.sum(self._dataset[:,self._col_assignment == i], axis = 1)
                        
        else:
            new_t = np.zeros((self._n_row_clusters, self._n_features), dtype = float)
            for i in range(self._n_row_clusters):
                new_t[i] = np.sum(self._dataset[self._row_assignment == i], axis = 0)
        return new_t


    def _perform_row_move(self):

        dataset, T = self._init_contingency_matrix(0)
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
       
        moves = 0
        for i in range(self._n_documents):
            all_tau = np.sum(dataset[i]*B, axis = 1)
            max_tau = np.max(all_tau)

            equal_solutions = np.where(max_tau == all_tau)[0]
            e_min = equal_solutions[0]
            if e_min != self._row_assignment[i]:
                moves += 1

            self._row_assignment[i] = e_min
        self._check_clustering(0)
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving rows, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")
        if moves ==0:
            return False
        else:
            return True


    def _perform_col_move(self):

        dataset, T = self._init_contingency_matrix(1)
        T = T.T
        dataset = dataset.T
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0
        for i in range(self._n_features):
            all_tau = np.sum(dataset[i]*B, axis = 1)
            max_tau = np.max(all_tau)

            equal_solutions = np.where(max_tau == all_tau)[0]
            e_min = equal_solutions[0]
            if e_min != self._col_assignment[i]:
                moves += 1
            self._col_assignment[i] = e_min

        self._check_clustering(1)
        if self.verbose:
            print(f"iteration {self._actual_n_iterations}, moving columns, n_clusters: ({self._n_row_clusters}, {self._n_col_clusters}), n_moves: {moves}")

        if moves ==0:
            return False
        else:
            return True

    def compute_taus(self):
        tot_per_x = np.sum(self._T, 1)
        tot_per_y = np.sum(self._T, 0)
        t_square = np.power(self._T, 2)

        a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
        b_x = np.sum(np.power(tot_per_x, 2))
        

        a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
        b_y = np.sum(np.power(tot_per_y, 2))

        tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
        tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))

        return tau_x, tau_y#, a_x, b_x


