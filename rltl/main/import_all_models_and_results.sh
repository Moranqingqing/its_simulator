#!/bin/bash

for xjobs in import_results #import_models #import_results
do
  for xenv in main_final #main_swap main_generalisation main_generalisation_stochastic main_contextual main_contextual_stochastic
  do
    echo "$xjobs $xenv"
	  python main.py configs/nicolas/main.yaml --override_config_file configs/nicolas/"$xenv".yaml --jobs "$xjobs"
  done
done
