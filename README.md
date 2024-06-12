# XCCSHAP
Source code for the paper "Combining SHAP-driven Clustering and Shallow Decision Trees to Explain XGBoost", submitted to Discovery Science 2024.

## How to reproduce the experiments:

Fine-tune XGBoost and a baseline Decision Tree (the latter is not really required):

```
sh ./run_uci_grid_search.sh
```

Run the experiments for XCCSHAP:

```
sh ./run_ccshap_uci_xccshap_load.sh
```

Run the experiments for MaSDT:

```
sh ./run_ccshap_uci_masdt_load.sh
```

Run the experiments for XGBTA:

```
sh ./run_ccshap_uci_xgbta_load.sh
```

The results are in the output directories, one file per dataset.