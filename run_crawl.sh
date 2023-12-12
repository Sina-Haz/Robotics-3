#!/bin/bash

for X in {0..4}; do
  for Y in {0..1}; do
    for Z in H L; do
        for N in H L; do
            python3 crawler.py --file estim1/estim1_${X}_${Y}_${Z}_${N}.npy --gt gts/gt_${X}_${Y}.npy
        done
    done 
  done
done