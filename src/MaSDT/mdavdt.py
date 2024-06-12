import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV

MAX_JOBS=16

parameters =  { 'dtc' : 
                  {'max_depth':           [2, 3, 4, 5, 10, 100]}
                }

def get_CV_model(X, y, model_type, parameters, scoring=None, n_folds=None, n_jobs=MAX_JOBS):
    model_cv = GridSearchCV(estimator=model_type, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=scoring, refit=True)
    model_cv.fit(X,y)
    best_model = model_cv.best_estimator_
    return best_model

def dist(x,y):
    return np.linalg.norm(x-y)
    #return scipy.spatial.distance.correlation(x,y)

def poprow(arr,i):
    pop = arr[i]
    new_array = np.vstack((arr[:i],arr[i+1:]))
    return new_array,pop

def cluster(X, p, k, dist_to_xr):
    #c = [p]
    #D = np.column_stack((X,[dist(v[:-1],p[:-1]) for v in X]))
    #D = D[D[:,-1].argsort()]
    #D = np.delete(D, -1, 1)
    #c.extend(D[:k-1])
    #D = D[k-1:]

    #xc = np.array([p[:-1] for p in c], copy=False, ndmin=2)
    #yc = np.array([p[-1] for p in c], copy=False)
    #cl = (xc, yc)
    #return D, cl

    c = [p]
    
    if dist_to_xr == None:
        distances = [dist(v[:-1],p[:-1]) for v in X]
    else:
        distances = dist_to_xr
        
    X = X[np.argpartition(distances, k-1)]
    c.extend(X[:k-1])
    X = X[k-1:]
    
    xc = np.array([p[:-1] for p in c], copy=False, ndmin=2)
    yc = np.array([p[-1] for p in c], copy=False)
    cl = (xc, yc)
    
    return X, cl
    
def mdav(X, y, k):
    D = np.column_stack((X,y))
    clusters = []
    while len(D) >= 3*k:
        # Centroid
        xm = np.mean(D, axis=0)
        # Furthest from centroid
        xri = np.argmax([dist(v[:-1],xm[:-1]) for v in D])
        D, xr = poprow(D, xri)
        # Furthest from furthest from centroid
        dist_to_xr = [dist(v[:-1],xr[:-1]) for v in D]
        xsi = np.argmax(dist_to_xr)
        dist_to_xr = dist_to_xr[:xsi]+dist_to_xr[xsi+1:]
        D, xs = poprow(D, xsi) 

        #cluster of xr
        D, c = cluster(D, xr, k, dist_to_xr)
        clusters.append(c)
        #cluster of xs
        D, c = cluster(D, xs, k, None)
        clusters.append(c)
        
    if len(D) >= 2*k and len(D) < 3*k:
        # Centroid
        xm = np.mean(D, axis=0)
        # Furthest from centroid
        xri = np.argmax([dist(v[:-1],xm[:-1]) for v in D])
        D, xr = poprow(D, xri)
        #cluster of xr
        D, c = cluster(D, xr, k, None)
        clusters.append(c)
        
        # rest of points
        xc = np.array([p[:-1] for p in D[:]], copy=False, ndmin=2)
        yc = np.array([p[-1] for p in D[:]], copy=False)
        cl = (xc, yc)
        clusters.append(cl)     
    else:
        # rest of points
        xc = np.array([p[:-1] for p in D[:]], copy=False, ndmin=2)
        yc = np.array([p[-1] for p in D[:]], copy=False)
        cl = (xc, yc)
        clusters.append(cl)
    
    centroids = np.array([np.mean(c[0],axis=0) for c in clusters], copy=False)
    
    return clusters, centroids

def gen_explanations(clustering, max_depth=-1):
    explanations = []
    for cluster in clustering:
        # Testing with max depth
        #if max_depth < 1:
        #    exp = tree.DecisionTreeClassifier()
        #else:
        #    exp = tree.DecisionTreeClassifier(max_depth=max_depth)
        #exp.fit(cluster[0],cluster[1])
        unique, counts = np.unique(cluster[1], return_counts=True)
        if min(counts)<2:
            if max_depth < 1:
                exp = tree.DecisionTreeClassifier()
            else:
                exp = tree.DecisionTreeClassifier(max_depth=max_depth)
            exp.fit(cluster[0],cluster[1])
        else:
            exp = get_CV_model(cluster[0], cluster[1], model_type=tree.DecisionTreeClassifier(), parameters=parameters['dtc'], n_folds=min(5,min(counts)))
        explanations.append(exp) 
    return explanations

def pre_explanations(explanations, centroids, X):
    predictions = []
    for sample in X:
        #select the closest classifier
        exp = explanations[np.argmin([dist(sample,c) for c in centroids])]
        exp_pred = exp.predict([sample])
        predictions.append(int(exp_pred[0]))
    return predictions

def pre_explanations_ext(explanations, centroids, X, T, n):
    predictions = []
    ret_exp = []
    ret_cen = []
    for sample, truth in zip(X,T):
        #select the 3 closest classifiers
        mins = np.array([dist(sample,c) for c in centroids]).argsort()[:n]
        for m in mins:
            exp = explanations[m]
            exp_pred = exp.predict([sample])
            if(exp_pred[0] == truth):
                break
        predictions.append(exp_pred[0])
        ret_exp.append(exp)
        ret_cen.append(centroids[m])
    return predictions, ret_exp, ret_cen

def mean_length_path_mdav(explanations, centroids, X):
    summat = []
    for sample in X:
        exp = explanations[np.argmin([dist(sample,c) for c in centroids])]
        pathmat = exp.decision_path([sample]).todense()
        summat.append(np.count_nonzero(pathmat)-1)
    mpl = np.mean(summat)
    stdpl = np.std(summat)
    return mpl, stdpl

def mean_length_path_mdav_ext(explanations, centroids, X, T, n):
    summat = []
    for sample, truth in zip(X,T):
        #select the 3 closest classifiers
        mins = np.array([dist(sample,c) for c in centroids]).argsort()[:n]
        for m in mins:
            exp = explanations[m]
            exp_pred = exp.predict([sample])
            pathmat = exp.decision_path([sample]).todense()
            if(exp_pred[0] == truth):
                break
        summat.append(np.count_nonzero(pathmat)-1)
    mpl = np.mean(summat)
    stdpl = np.std(summat)
    return mpl, stdpl