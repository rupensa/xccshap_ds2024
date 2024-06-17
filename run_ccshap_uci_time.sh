#!/bin/bash

mkdir ./outuci_time

python ./src/expe_uci_classification_xccshap.py --id=2 --modelpath xgb_model --out outuci_time --classifier=xgb
python ./src/expe_uci_classification_masdt.py --id=2 --modelpath xgb_model --out outuci_time --classifier=xgb
python ./src/expe_uci_classification_xgbta.py --id=2 --modelpath xgb_model --out outuci_time --classifier=xgb

