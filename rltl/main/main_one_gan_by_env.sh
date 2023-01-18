#!/bin/bash

gan_single_envs=()
for i in `seq 0 50`;
do
  gan_single_envs+=( "Gridworld_sources-$i" )
  gan_single_envs+=( "Gridworld_targets-$i" )
done

for i in "${gan_single_envs[@]}"; do
  echo "PYTHONPATH=\"$PYTHONPATH:$4\" $3 main.py $1 --jobs learn_gans --gan_baselines one_gan_by_env --gan_single_envs $i --show_plots False --override_config_file $2 &"
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
done
