#!/bin/bash



mkdir ./outuci_masdt

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_masdt.py --id=$p --modelpath xgb_model --out outuci_masdt --classifier=xgb;
done <datasets_xgb_all.txt

