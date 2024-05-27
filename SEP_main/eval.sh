#!/bin/bash

for DATASET in sun397 caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft dtd eurosat  ucf101
do
python parse_test_res.py   output_0514_6_0_6_8/base2new/train_base/${DATASET}/shots_16_8.0/TCP/vit_b16_ep100_ctxv1/ --test-log
python parse_test_res.py   output_0514_6_0_6_8/base2new/test_new/${DATASET}/shots_16_8.0/TCP/vit_b16_ep100_ctxv1/ --test-log
done
