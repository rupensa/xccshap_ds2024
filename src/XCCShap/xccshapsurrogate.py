import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree
import pydotplus
from sklearn import tree
import sys

MAX_JOBS = 1
parameters =  { 'dtc' : 
                  {'max_depth':           [2, 3, 4, 5, 10, 100]}
                }

class XCCShapSurrogate():

    def __init__(self, explainer_model, max_depth=None, topk=None, n_jobs = MAX_JOBS, random_state=None):
        self.n_jobs=n_jobs
        self.topk=topk
        self.random_state=random_state
        self.max_depth = max_depth
        self.dt_model = DecisionTreeClassifier()
        if random_state is None:
            self.random_state = np.random.RandomState()
        self.explainer_model=explainer_model
        self.row_labels=self.explainer_model.row_labels_
        self.col_labels=self.explainer_model.col_labels_
        self.n_models = len(np.unique(self.row_labels, return_counts=False).astype(int))
        self.shapmat=self.explainer_model.shapmat 
        self.model=[]
        if not(type(self.explainer_model).__name__ in ['XCCShap']):
            sys.exit('Error: model type not supported.')

    def get_CV_model(self, X, y, model_type, parameters, scoring=None, n_folds=None, n_jobs=MAX_JOBS):
        model_cv = GridSearchCV(estimator=model_type, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=scoring, refit=True)
        model_cv.fit(X,y)
        best_model = model_cv.best_estimator_
        return best_model
        
    def _gini(self, x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x)**2 * np.mean(x))

    def fit(self, X, y):
        clust_val = {}
        clust_lab = np.unique(self.row_labels, return_counts=False).astype(int)
        clust_lab_col = np.unique(self.col_labels, return_counts=False).astype(int)
        state = {}
        self.model_features={}
        new_state = {}
        values = np.empty((0,1,len(np.unique(y.to_numpy()))))
        classes = np.unique(y.to_numpy())
        new_state['max_depth'] = 0
        new_state['node_count'] = 0
        offset = 0
        lev=0
        for k in clust_lab:
            self.model_features[k]=np.array([])
            for l in clust_lab_col:
                clust_val[k,l]=self.shapmat[self.row_labels==k,:][:,self.col_labels==l].mean(axis=0)
                if not (self.topk is None):
                    topk=min(self.topk,np.shape(self.shapmat[self.row_labels==k,:][:,self.col_labels==l])[1])
                    feat = np.where(self.col_labels==l)[0][np.argpartition(clust_val[k,l], -topk)[-topk:]]
                    self.model_features[k]=np.append(self.model_features[k],feat)
                else:
                    feat = np.where(self.col_labels==l)[0][np.where([(clust_val[k,l]>=clust_val[k,l].mean()) & (clust_val[k,l]>0)])[1]]
                    self.model_features[k]=np.append(self.model_features[k],feat)
            self.model_features[k].sort()
            unique, counts = np.unique(y.iloc[self.row_labels==k], return_counts=True)
            if min(counts)<2:
                model = DecisionTreeClassifier(random_state=self.random_state)
                model.fit(X.iloc[self.row_labels==k,self.model_features[k]], y.iloc[self.row_labels==k])
                self.model.append(model)
            else:
                self.model.append(self.get_CV_model(X.iloc[self.row_labels==k,self.model_features[k]], y.iloc[self.row_labels==k], model_type=DecisionTreeClassifier(random_state=self.random_state), parameters=parameters['dtc'], n_folds=min(5,min(counts))))
            state[k]=self.model[k].tree_.__getstate__()
            missing_classes = np.setdiff1d(classes, self.model[k].classes_)
            if len(missing_classes)>0:
                corrected_values = np.empty((0,1,len(np.unique(y.to_numpy()))), dtype=int)
                for item in range(len(state[k]["values"])):
                    tmp_item = np.empty(len(np.unique(y.to_numpy())), dtype=int)
                    tmp_item[self.model[k].classes_] = state[k]["values"][item]
                    tmp_item[missing_classes] = np.zeros(len(missing_classes), dtype=int)
                    corrected_values=np.append(corrected_values, np.array([[tmp_item]]), axis=0)
                state[k]["values"]=corrected_values
            if (k < (self.n_models-1)):
                lev+=1
                if not(k):
                    node_dt = state[k]['nodes'].dtype
                    node = np.empty(self.n_models-1,dtype=node_dt)
                node[k]['left_child'] = self.n_models + offset - 1
                if (k==(self.n_models-2)):
                    node[k]['right_child'] = self.n_models + offset + state[k]['node_count'] - 1
                else:
                    node[k]['right_child'] = k+1
                node[k]['feature'] = len(X.columns)
                node[k]['threshold'] = k
                node[k]['impurity'] = self._gini(y.iloc[self.row_labels==k].to_numpy())
                node[k]['n_node_samples'] = np.shape(X)[0]
                for i in range(k):
                    node[k]['n_node_samples'] -= np.shape(X.iloc[self.row_labels==(k-1)])[0]
                node[k]['weighted_n_node_samples'] = float(node[k]['n_node_samples'])
                node[k]['missing_go_to_left'] = 1
                if (k):    
                    res = {key: y.iloc[self.row_labels==(k-1)].values.tolist().count(key) for key in classes.tolist()}
                    class_count=list(res.values())
                else:
                    res = {key: y.values.tolist().count(key) for key in classes.tolist()}
                    class_count=list(res.values())
                class_count=np.array([[class_count]])
                values = np.append(values, class_count, axis=0)
                for i in range(state[k]['node_count']):
                    state[k]['nodes']['feature'][i]=self.model_features[k][state[k]['nodes']['feature'][i]]
                    if state[k]['nodes']['left_child'][i]>=0:
                        state[k]['nodes']['left_child'][i] += node[k]['left_child']
                    if state[k]['nodes']['right_child'][i]>=0:
                        state[k]['nodes']['right_child'][i] += node[k]['left_child']
            else:
                for i in range(state[k]['node_count']):
                    state[k]['nodes']['feature'][i]=self.model_features[k][state[k]['nodes']['feature'][i]]
                    if state[k]['nodes']['left_child'][i]>=0:
                        state[k]['nodes']['left_child'][i] += node[k-1]['right_child']
                    if state[k]['nodes']['right_child'][i]>=0:
                        state[k]['nodes']['right_child'][i] += node[k-1]['right_child']
            new_state['max_depth'] = max(new_state['max_depth'], (state[k]['max_depth']+lev))
            offset += state[k]['node_count']
        new_state['node_count'] = offset + self.n_models - 1
        new_state['nodes']=node
        new_state['values']=values
        for k in clust_lab:
            new_state['nodes'] = np.append(new_state['nodes'], state[k]['nodes'], axis=0)
            new_state['values'] = np.append(new_state['values'], state[k]['values'], axis=0)
        new_tree=Tree(np.shape(X)[1]+1, np.array([len(classes)]), 1)
        new_tree.__setstate__(new_state)
        self.dt_model.tree_=new_tree
        self.dt_model.n_outputs_=1
        self.dt_model.classes_=classes
        self.dt_model.n_features_in_=len(X.columns)+1

    def predict(self, V):
        shapmat = self.explainer_model._get_shapmat(V)
        labels =  self.explainer_model.xccmodel.assign_samples(shapmat)
        V = np.append(V, labels, axis=1)
        return self.dt_model.predict(V)
    
    def decision_path(self, V):
        shapmat = self.explainer_model._get_shapmat(V)
        labels =  self.explainer_model.xccmodel.assign_samples(shapmat)
        V = np.append(V, labels, axis=1)
        return self.dt_model.decision_path(V)
    
    def plot_tree(self, **kwargs):
        if 'feature_names' in kwargs:
            feature_names=kwargs.pop('feature_names')
            feature_names=np.concatenate((feature_names, ['XCCSCluster']), axis=0)
        else:
            feature_names=None
        tree.plot_tree(self.dt_model, feature_names=feature_names, **kwargs)        

    def plot_decision_path(self, decision_paths, filename=None, feature_names=None, class_names=None):
        feature_names=np.concatenate((feature_names, ['XCCSCluster']), axis=0)
        dot_data = tree.export_graphviz(self.dt_model, out_file=filename,
                                        feature_names=feature_names,
                                        class_names=class_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)

        for node in graph.get_node_list():
            if node.get_attributes().get('label') is None:
                continue
            if 'samples = ' in node.get_attributes()['label']:
                labels = node.get_attributes()['label'].split('<br/>')
                for i, label in enumerate(labels):
                    if label.startswith('samples = '):
                        labels[i] = 'samples = 0'
                node.set('label', '<br/>'.join(labels))
                node.set_fillcolor('white')

        for decision_path in decision_paths:
            for n, node_value in enumerate(decision_path.toarray()[0]):
                if node_value == 0:
                    continue
                node = graph.get_node(str(n))[0]            
                node.set_fillcolor('green')
                labels = node.get_attributes()['label'].split('<br/>')
                for i, label in enumerate(labels):
                    if label.startswith('samples = '):
                        labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

                node.set('label', '<br/>'.join(labels))

        if (filename is None):
            filename = 'tree.png'
        graph.write_png(filename)

        

