import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
import sys, getopt
from XGBoostTreeApproximator.FBT import FBT
import expe_utils as ccshap
import os.path
import warnings
import os

rng = np.random.RandomState(42)
RANDOM_SEED = 42
MAX_JOBS=16
os.environ["TQDM_DISABLE"] = "1"


def mean_length_path(model, X, n_features):
   idx = np.where(model.dt_model.tree_.__getstate__()['nodes']['feature']>=n_features)[0]
   pathmat = np.array(model.decision_path(X).todense())
   summat = np.count_nonzero(pathmat,axis=1)-np.count_nonzero(pathmat[:,idx],axis=1)
   mpl = np.mean(summat)
   stdpl = np.std(summat)
   return mpl, stdpl

def mean_length_path_baseline(model, X):
   pathmat = np.array(model.decision_path(X).todense())
   summat = np.count_nonzero(pathmat,axis=1)
   mpl = np.mean(summat)
   stdpl = np.std(summat)
   return mpl, stdpl

def mean_length_path_fbt(model, X):
   dplist = model.get_decision_paths(X)
   dparray = np.zeros(len(dplist))
   for i in range(len(dplist)):
      dparray[i] = len(dplist[i])-1
   mpl = np.mean(dparray)
   stdpl = np.std(dparray)
   return mpl, stdpl

def gridsearchFBT(X,y,class_model,traincol,labelcol,parameters):
   n_splits = min(5,len(X.to_numpy()))
   kf = KFold(n_splits=n_splits, random_state=rng, shuffle=True)
   max_score = -1.0
   for param in parameters:
      scores = np.zeros(n_splits)
      for i, (train_index, test_index) in enumerate(kf.split(X)):
         model=FBT(max_depth=param,max_number_of_conjunctions=100,min_forest_size=10,pruning_method='auc')
         train = pd.merge(X.iloc[train_index], y.iloc[train_index], left_index=True, right_index=True)
         model.fit(train,traincol,labelcol,class_model)
         y_predicted=model.predict(X.iloc[test_index])
         scores[i] = accuracy_score(y.iloc[test_index].to_numpy(),y_predicted)
      if np.mean(scores)>max_score:
         max_score = np.mean(scores)
         best_param = param
   best_model = FBT(max_depth=best_param,max_number_of_conjunctions=100,min_forest_size=10,pruning_method='auc')
   train = pd.merge(X, y, left_index=True, right_index=True)
   best_model.fit(train,traincol,labelcol,class_model)
   return best_model


def ccshap_full(id_dataset, output_path, model_path, max_depth=100, model_type = XGBClassifier()):
   df_X, df_y, dataset_name, cat_features = ccshap.open_preprocess_uci_dataset(id_dataset)
   print(dataset_name)
   for target_col in df_y.columns:
      class_acc = []
      class_f1 = []
      class_mcc = []
      surr_test_acc = []
      surr_test_f1 = []
      surr_test_mcc = []
      surr_test_avg_pl = []
      surr_test_std_pl = []
      nclass = []
      dsname = []
      dstarget = []
      nrows = []
      ncols = []
      nrows_train = []
      ncols_train = []
      nrows_test = []
      ncols_test = []
      print('Using target: ',target_col)
      df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y[target_col], test_size=0.3, random_state=rng)
      df_X_all = df_X.to_numpy()
      print('Loading model...')
      model_file_name = "dataset_"+str(id_dataset)+"_"+target_col.replace(" ", "_")
      model_file = os.path.join(model_path, model_file_name)
      class_model = ccshap.load_model(model_type, model_file)
      if class_model is None:
         continue
      y_predicted = class_model.predict(df_X_test)
      class_acc.append(accuracy_score(df_y_test.to_numpy(),y_predicted))
      class_f1.append(f1_score(df_y_test.to_numpy(),y_predicted, average='macro'))
      class_mcc.append(matthews_corrcoef(df_y_test.to_numpy(),y_predicted))
      print('Computing surrogate models...')
      surr_model = gridsearchFBT(df_X_train,df_y_train,class_model,df_X_train.columns,target_col,parameters=[3,4,5,10,100])
      y_test_predicted_surr=surr_model.predict(df_X_test)
      surr_test_acc.append(accuracy_score(y_predicted,y_test_predicted_surr))
      surr_test_f1.append(f1_score(y_predicted,y_test_predicted_surr, average='macro'))
      surr_test_mcc.append(matthews_corrcoef(y_predicted,y_test_predicted_surr))
      mpl, stdpl = mean_length_path_fbt(surr_model, df_X_test)
      surr_test_avg_pl.append(mpl)
      surr_test_std_pl.append(stdpl)


      nclass.append(df_y[target_col].nunique())
      dsname.append(dataset_name)
      dstarget.append(target_col)
      nrows.append(np.shape(df_X_all)[0])
      ncols.append(np.shape(df_X_all)[1])
      nrows_train.append(np.shape(df_X_train)[0])
      ncols_train.append(np.shape(df_X_train)[1])
      nrows_test.append(np.shape(df_X_test)[0])
      ncols_test.append(np.shape(df_X_test)[1])
      data = {}
      data["dataset"] = dsname
      data["class"] = dstarget
      data["#class"] = nclass
      data["#rows"] = nrows
      data["#cols"] = ncols
      data["#rows_train"] = nrows_train
      data["#cols_train"] = ncols_train
      data["#rows_test"] = nrows_test
      data["#cols_test"] = ncols_test
      data["class_acc"] = class_acc
      data["class_f1"] = class_f1
      data["class_mcc"] = class_mcc
      data["surr_test_acc"] = surr_test_acc
      data["surr_test_f1"] = surr_test_f1
      data["surr_test_mcc"] = surr_test_mcc
      data["surr_test_avg_pl"] = surr_test_avg_pl
      data["surr_test_std_pl"] = surr_test_std_pl
      print('Done.')
      out_table=pd.DataFrame(data)
      output_file_name = "xccshap_full_fbt_"+dataset_name.replace(" ", "_")+"_"+target_col.replace(" ", "_")+".csv"
      output_file = os.path.join(output_path, output_file_name)
      out_table=pd.DataFrame(data)
      out_table.to_csv(output_file, index=False)

def main(argv):
   output_path = '.'
   id_dataset = 53
   classifier = 'xgb'
   model_path = '.'
   max_depth = 100
   warnings.filterwarnings('ignore') 
   opts, args = getopt.getopt(argv,"hi:o:",["help","out=","id=","max_depth=", "modelpath=","classifier="])
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print ('ccshap_expe.py --out <outputdir> []')
         sys.exit()
      elif opt in ("-o", "--out"):
         output_path = arg
      elif opt in ("-m", "--modelpath"):
         model_path = arg
      elif opt in ("-m", "--max_depth"):
         max_depth = int(arg)
      elif opt in ("-c", "--classifier"):
         if not(arg in ['xgb']):
            sys.exit('Error: classifier model type not supported.')
         classifier = arg
      elif opt in ("-i", "--id"):
         id_dataset = int(arg)
   print("CCSHAP: test using "+classifier+" on: "+str(id_dataset))
   if (classifier=='xgb'):
      ccshap_full(id_dataset, output_path, model_path, max_depth=max_depth, model_type=XGBClassifier(verbosity=0,  tree_method='hist', device='cuda'))
if __name__ == "__main__":
   main(sys.argv[1:])