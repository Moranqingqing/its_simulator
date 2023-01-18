#!/bin/bash

for xenv in main_swap main_generalisation main_generalisation_stochastic main_contextual main_contextual_stochastic
do
  echo "$xjobs $xenv"
  python main.py configs/nicolas/main.yaml --override_config_file configs/nicolas/"$xenv".yaml --jobs "learn_gans" --gan_baselines "$1" &
done
