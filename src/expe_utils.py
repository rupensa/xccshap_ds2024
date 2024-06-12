import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
from joblib import dump, load
import pickle
import os
import re

MAX_JOBS = 16

def save_uci_dataset(id_dataset, path="."):
    data_to_save = {}
    output_file = os.path.join(path, f'uci_dataset_{id_dataset}.pkl')
    dataset = fetch_ucirepo(id=id_dataset)
    data_to_save['features']=dataset.data.features
    data_to_save['targets']=dataset.data.target
    data_to_save['name']=dataset.metadata.name
    with open(output_file, 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def open_preprocess_uci_dataset(id_dataset, cat_encode=True, missing_col_treshold=0.5, encode_labels=True):
    output_file = os.path.join("./uci_dataset", f'uci_dataset_{id_dataset}.pkl')
    with open(output_file, 'rb') as f:
        dataset = pickle.load(f)
    df_X = dataset['features']
    df_y = dataset['targets']
    target_columns = df_y.columns
    df_temp = df_X.merge(df_y, left_index=True, right_index=True)
    df_temp = df_temp.dropna(thresh=missing_col_treshold*len(df_temp), axis=1)
    df_temp = df_temp.dropna()
    df_X = df_temp.drop(target_columns, axis=1)
    df_y = df_temp[target_columns]
    categorical_ix = df_X.select_dtypes(include=['object', 'boolean', 'string', 'category']).columns
    if (cat_encode):
        df_X = pd.get_dummies(df_X, columns=categorical_ix, dtype='float')
    else:
        df_X[categorical_ix] = df_X[categorical_ix].astype('category')
    if (encode_labels):
        df_y = df_y.apply(preprocessing.LabelEncoder().fit_transform)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    df_X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<', '?'))) else col for col in df_X.columns.values]
    return  df_X, df_y, dataset['name'], categorical_ix

def get_classifier(X, y, model_type = RandomForestClassifier(n_estimators=100, random_state=None, bootstrap=True, n_jobs=MAX_JOBS)):
    model = model_type
    model.fit(X.to_numpy(), y.to_numpy())
    return model
    
def get_CV_model(X, y, model_type, parameters, scoring=None, n_folds=None, n_jobs=MAX_JOBS):
    model_cv = GridSearchCV(estimator=model_type, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=scoring, refit=True)
    model_cv.fit(X,y)
    best_model = model_cv.best_estimator_
    return best_model

def get_CV_model_opt(X, y, model_type, parameters, scoring=None, n_folds=None, n_jobs=MAX_JOBS):
    cv=StratifiedKFold(n_splits=n_folds)
    model_cv = HalvingGridSearchCV(estimator=model_type, param_grid=parameters, cv=cv, n_jobs=n_jobs, scoring=scoring, min_resources='smallest', max_resources='auto', refit=True)
    model_cv.fit(X,y)
    best_model = model_cv.best_estimator_
    return best_model

def save_model(model, filename):
    if (type(model).__name__ in ['XGBClassifier']):
        model.save_model(filename+".json")
    else:
        dump(model, filename+".joblib") 

def load_model(model_type, filename):
    if (type(model_type).__name__ in ['XGBClassifier']):
        model = XGBClassifier()
        try:
            model.load_model(fname = filename+".json")
        except:
            print('Error while opening file: '+filename+'.json')
            model = None
    else:
        try:
            model = load(filename+".joblib")
        except:
            print('Error while opening file: '+filename+'.joblib')
            model = None
    return model 


