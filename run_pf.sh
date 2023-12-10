#!/bin/bash

for X in {0..4}; do
  for Y in {0..1}; do
    for Z in H L; do
        for N in H L; do
            if [ "$N" == "H" ]; then
                num_particles=2000
            elif [ "$N" == "L" ]; then
                num_particles=500
            fi
            python3 particle_filter.py --map maps/landmarks_${X}.npy \
                                --sensing readings/readings_${X}_${Y}_${Z}.npy \
                                --num_particles $num_particles \
                                --estimates estim1/estim1_${X}_${Y}_${Z}_${N}.npy
        done
    done 
  done
done