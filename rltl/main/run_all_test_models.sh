#!/bin/bash

k=0
pids=()

for xenv in main_swap main_generalisation main_generalisation_stochastic main_contextual main_contextual_stochastic main_final
do
  echo "$xjobs $xenv"
  python main.py configs/nicolas/main.yaml --override_config_file configs/nicolas/"$xenv".yaml --jobs "inferences" &
  pids[${k}]=$!
  k=$((k + 1))
done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

for xenv in main_swap main_generalisation main_generalisation_stochastic main_contextual main_contextual_stochastic main_final
do
  echo "$xjobs $xenv"
  python main.py configs/nicolas/main.yaml --override_config_file configs/nicolas/"$xenv".yaml --jobs "print_test_models_results" &
done