#!/bin/bash
rm -f result/*
#python gen_result.py --data_dir=../model/fractor_0.2/inference/ctw_test/
python gen_result.py #--data_dir=../../model/ART/inference/ART_val/
#python sortdetection.py
#python test_ctw1500_eval.py
