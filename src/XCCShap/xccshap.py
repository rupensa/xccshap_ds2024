import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from fasttreeshap import TreeExplainer
import matplotlib.pyplot as plt
from tqdm import tqdm
from XCCShap.ccutils import XCoClust, CoClust
from time import time

MAX_JOBS = 1
NCCRUNS = 30

class XCCShap():

    def __init__(self, model, explainer='tree', method='max', data=None, feature_perturbation="tree_path_dependent", normalize=True, topk=None, n_jobs = MAX_JOBS, nccrun=NCCRUNS, ccprogressbar=False, **kwargs):
        self.method = method
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.topk = topk
        self.model = model
        self.nccrun=nccrun
        self.ccprogressbar=ccprogressbar
        if explainer == 'tree':
            if not(type(self.model).__name__ in ['RandomForestClassifier', 'RandomForestRegressor', 'DecisionTreeClassifier', 'DecisionTreeRegressor', 'XGBClassifier', 'XGBRegressor', 'CatBoostClassifier', 'CatBoostRegressor']):
                raise ValueError('Error: model type not supported.')
            self.explainer = TreeExplainer(self.model, data, feature_perturbation=feature_perturbation, algorithm = "auto", n_jobs = self.n_jobs, check_additivity=False, silent=True, **kwargs)
        else:
            raise ValueError('Unsupported explainer. Only the following explainer classes are supported: \'tree\', \'kernel\' and \'gpu\'.')



    def explain(self, X):
        self.shapmat=self._get_shapmat(X)
        self.row_labels_, self.col_labels_, self.tau_x_, self.tau_y_ = self._get_shap_coclustering()
        return self.shapmat, self.row_labels_, self.col_labels_

    def _shap_norm(self, arr):
        shapvec = arr/np.max(arr)
        return shapvec

    def _get_shapmat(self, X):
        y = self.model.predict(X)
        start_time = time()
        if (type(self.explainer).__name__ in ['KernelExplainer']):
            shapmat = self.explainer.shap_values(X, silent=True)
        else:
            shapmat = self.explainer.shap_values(X, check_additivity=False)
        end_time = time()
        print(f'SHAP computation time: {end_time-start_time}')
        if (len(np.shape(shapmat)) > 2):
            if (self.method=='perclass'):
                shapmat = abs(shapmat[y[0]])
            else:
                shapmat = abs(np.max(shapmat,axis=0))
        else:
                shapmat = abs(shapmat)  
        if (self.normalize):
            shapmat = np.apply_along_axis(self._shap_norm, axis=1, arr=shapmat)
        if not (self.topk is None):
            shapmat = self._truncate_top_k(shapmat, k=self.topk)
        return shapmat

    def _get_shap_coclustering(self):
        model = {}
        tauxy = {}
        data=self.shapmat
        emptycols = np.where(~data.any(axis=0))[0]
        data=np.delete(data, emptycols, axis=1)
        for run in tqdm(range(self.nccrun), disable=not(self.ccprogressbar)):
            model[run] = CoClust(initialization = 'extract_centroids', k=min(50,np.shape(data)[0]), l=min(50,np.shape(data)[1]), verbose = False)
            model[run].fit(data)
            tau_x, tau_y = model[run].compute_taus()
            tauxy[run] = tau_x
        bestrun = max(tauxy, key=tauxy.get)
        new_clust_label = max(model[bestrun].column_labels_) + 1
        row_labels=np.array(model[bestrun].row_labels_).astype(int)
        col_labels=np.array(model[bestrun].column_labels_).astype(int)
        tau_x, tau_y = model[bestrun].compute_taus()
        for i in emptycols:
            col_labels = np.insert(col_labels,i,new_clust_label)
        clust_labels, clust_counts = np.unique(row_labels, return_counts=True)
        majority_clust = np.argmax(clust_counts)
        row_labels[np.where(np.isin(row_labels,clust_labels[clust_counts < 3]))[0]] = majority_clust
        le = LabelEncoder()
        row_labels=le.fit_transform(row_labels)
        col_labels=le.fit_transform(col_labels)
        self.xccmodel = XCoClust(X=self.shapmat, k=min(50,np.shape(data)[0]), l=min(50,np.shape(data)[1]), row_labels=row_labels, col_labels=col_labels)
        return row_labels, col_labels, tau_x, tau_y
    
    def _truncate_top_k(self, x, k, inplace=False):
        m, n = x.shape
        topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
        rows, _ = np.indices((m, k))
        kth_vals = x[rows, topk_indices].min(axis=1)
        is_smaller_than_kth = x < kth_vals[:, None]
        if not inplace:
            return np.where(is_smaller_than_kth, 0, x)
        x[is_smaller_than_kth] = 0
        return x

    def plot_reorganized_matrix(self, precision=0.2, markersize=0.9):
        plt.style.use('seaborn-muted') 
        row_indices = np.argsort(self.row_labels)
        col_indices = np.argsort(self.col_labels)
        X_reorg = self.shapmat[row_indices, :]
        X_reorg = X_reorg[:, col_indices]
        plt.title('Coclustering results')
        plt.spy(X_reorg, precision=precision, markersize=0.3, aspect='auto', color='black')
        self._remove_ticks()
        plt.show()

    def _remove_ticks(self):
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        right='off', left='off')

    def plot_cc_distribution(self):
        unique, counts = np.unique(self.row_labels, return_counts=True)
        plt.bar(unique, counts, width=0.8, bottom=None, align='center')
        plt.title('Row clusters distribution')
        plt.show()
        unique, counts = np.unique(self.col_labels, return_counts=True)
        plt.bar(unique, counts, width=0.8, bottom=None, align='center')
        plt.title('Column clusters distribution')
        plt.show()

    def plot_shap_coclusters(self, labels=None):
        idx_labels = np.arange(np.shape(self.shapmat)[1])
        clust_lab = np.unique(self.row_labels_, return_counts=False).astype(int)
        clust_lab_col = np.unique(self.col_labels_, return_counts=False).astype(int)
        fig, axs = plt.subplots(np.max(clust_lab)+1, np.max(clust_lab_col)+1)
        fig.tight_layout()
        fig.set_figwidth(5*np.max(clust_lab_col))
        fig.set_figheight(5*np.max(clust_lab))
        for k in clust_lab:
            for l in clust_lab_col:
                if not (labels is None):
                    sublabels = labels.take(np.where(self.col_labels_==l))[0]
                else:
                    sublabels = idx_labels[np.array(self.col_labels_)==l]
                axs[k,l].boxplot(self.shapmat[self.row_labels_==k][:,np.array(self.col_labels_)==l], labels=sublabels)
                axs[k,l].set_title('Cocluster no. ('+str(k)+','+str(l)+')')
                axs[k,l].set_xticklabels(sublabels, rotation=45, ha="right")
                ##axs[k,l].xticks(rotation=45, ha="right")
        plt.subplots_adjust(hspace=0.9)
        plt.show()

    def plot_shap_all(self, labels=None):
        plt.boxplot(self.shapmat,labels=labels)
        plt.xticks(rotation=45, ha="right")
        plt.show()