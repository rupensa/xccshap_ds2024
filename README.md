# XCCSHAP
Source code for the paper "Combining SHAP-Driven Co-clustering and Shallow Decision Trees to Explain XGBoost", Proceedings of 27th International Conference on Discovery Science 2024. Lecture Notes in Computer Science, vol 15243. Springer, Cham, by Pensa, R.G., Crombach, A., Peignier, S., Rigotti, C. [[paper]](https://doi.org/10.1007/978-3-031-78977-9_24)

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

Run the experiments to get the runtime on Adult:

```
sh ./run_ccshap_uci_time.sh
```

The results are in the output directories, one file per dataset.

## How to cite:
```
@inproceedings{PensaCPR25,
  author       = {Pensa, Ruggero G. and Crombach, Anton and Peignier, Sergio and Rigotti, Christophe},
  editor       = {Pedreschi, Dino and Monreale, Anna and Guidotti, Riccardo and Pellungrini, Roberto and Naretto, Francesca},
  title        = {Parameter-Less Tensor Co-clustering},
  booktitle    = {Discovery Science - 27th International Conference, {DS} 2024, Pisa,
                  Italy, October 14-16, 2024, Proceedings},
  series       = {Lecture Notes in Computer Science},
  volume       = {15243},
  pages        = {369--384},
  publisher    = {Springer},
  year         = {2025}
}
```
