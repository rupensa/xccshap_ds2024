#!/bin/bash

mkdir ./xgb_model
mkdir ./dtc_model

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_gridsearch.py --id=$p --out xgb_model --classifier=xgb;
  python ./src/expe_uci_gridsearch.py --id=$p --out dtc_model --classifier=dtc;
done <datasets_xgb_all.txt
