import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys, getopt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import expe_utils as ccshap
import os.path
import warnings
import os
import json

rng = np.random.RandomState(42)
RANDOM_SEED = 42
MAX_JOBS=16
os.environ["TQDM_DISABLE"] = "1"

parameters =  { 'dtc' : 
                  {'max_depth':           [2, 3, 4, 5, 10, 100],
                   'min_samples_split':   [1, 2, 3, 4, 5, 10, 100],
                   'min_samples_leaf':    [1, 2, 3, 4, 5, 10, 100]
                   },
               'xgb' :
                  {
                  'min_child_weight':  [1, 5, 10],
                  'gamma':             [0.5, 1, 1.5, 2],
                  'n_jobs':            [MAX_JOBS],
                  'random_state':      [rng],
                  'max_depth':         [3, 4, 5]
                  }
               }


def ccshap_full(id_dataset, output_path, parameters, model_type = RandomForestClassifier(), encode_labels=True):
   df_X, df_y, dataset_name, cat_features = ccshap.open_preprocess_uci_dataset(id_dataset, encode_labels=encode_labels)
   print(dataset_name)
   for target_col in df_y.columns:
      print('Using target: ',target_col)
      df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y[target_col], test_size=0.3, random_state=rng)
      df_X_all = df_X.to_numpy()
      print('Tuning model...')
      if (np.shape(df_X_train)[0]<1000):
         class_model = ccshap.get_CV_model(df_X_train, df_y_train, model_type, parameters, n_folds=5)   
      else:
         class_model = ccshap.get_CV_model_opt(df_X_train, df_y_train, model_type, parameters, n_folds=5)
      y_predicted = class_model.predict(df_X_test)
      print('Done.')
      output_file_name = "dataset_"+str(id_dataset)+"_"+target_col.replace(" ", "_")
      output_file = os.path.join(output_path, output_file_name)
      ccshap.save_model(class_model,output_file)

def main(argv):
   output_path = ''
   id_dataset = 53
   classifier = 'xgb'
   params_file = None
   warnings.filterwarnings('ignore') 
   opts, args = getopt.getopt(argv,"hi:o:",["help","out=","id=","params=","classifier="])
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print ('ccshap_expe.py --out <outputdir> []')
         sys.exit()
      elif opt in ("-o", "--out"):
         output_path = arg
      elif opt in ("-p", "--params"):
         params_file = arg
      elif opt in ("-c", "--classifier"):
         if not(arg in ['xgb','dtc']):
            sys.exit('Error: classifier model type not supported.')
         classifier = arg
      elif opt in ("-i", "--id"):
         id_dataset = int(arg)
   print("CCSHAP: grid search using "+classifier+" on: "+str(id_dataset))
   if not (params_file is None):
      with open(params_file) as json_file:
         parameters_grid = json.load(json_file)
   else:
      parameters_grid = parameters[classifier]
   if (classifier=='xgb'):
      ccshap_full(id_dataset, output_path, parameters=parameters_grid, model_type=XGBClassifier(verbosity=0, tree_method='hist', device='cuda'))
   elif (classifier=='dtc'):
      ccshap_full(id_dataset, output_path, parameters=parameters_grid, model_type=DecisionTreeClassifier())
if __name__ == "__main__":
   main(sys.argv[1:])