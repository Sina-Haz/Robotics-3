#!/bin/bash

for X in {0..4}; do
  for Y in {0..1}; do
    for Z in H L; do
      python3 simulate.py --plan controls/controls_${X}_${Y}.npy \
                          --map maps/landmarks_${X}.npy \
                          --execution gts/gt_${X}_${Y}.npy \
                          --sensing readings/readings_${X}_${Y}_${Z}.npy
    done
  done
done