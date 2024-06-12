#!/bin/bash


mkdir ./outuci_xsshap

while read p;
do
  echo Processing $p;
  python ./src/expe_uci_classification_xccshap.py --id=$p --modelpath xgb_model --out outuci_xsshap --classifier=xgb;
done <datasets_xgb_all.txt

