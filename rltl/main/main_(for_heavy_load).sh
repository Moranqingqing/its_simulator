#!/bin/bash

gan_single_envs=()
#sources1=()
#sources2=()
#sources3=()
#sources4=()
sources=()

targets=()
#targets1=()
#targets2=()
for i in `seq 0 13`;
do
  gan_single_envs+=( "Gridworld_sources-$i" )
  sources+=( "Gridworld_sources-$i" )
done

for i in `seq 0 5`;
do
  gan_single_envs+=( "Gridworld_targets-$i" )
  targets+=( "Gridworld_targets-$i" )
done
#
#for i in `seq 0 4`;
#do
#  sources1+=( "Gridworld_sources-$i" )
#done
#
#for i in `seq 5 9`;
#do
#  sources2+=( "Gridworld_sources-$i" )
#done
#
#for i in `seq 10 13`;
#do
#  sources3+=( "Gridworld_sources-$i" )
#done
#
#for i in `seq 0 4`;
#do
#  targets1+=( "Gridworld_targets-$i" )
#done
#
#for i in `seq 5 9`;
#do
#  targets2+=( "Gridworld_targets-$i" )
#done

# ========================================================
# ========================================================
# ========================================================
k=0
pids=()

for i in "${gan_single_envs[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs create_datasets --envs_to_generate "$i" --show_plots False --override_config_file "$2" &
  pids[${k}]=$!
  k=$((k + 1))
done


# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done



echo "datasets done"


# ========================================================
# ========================================================
# ========================================================

classifier_baselines=(
  "full_bayesian"
  "feedforward"
)

k=0
pids=()

 ========================================================
 ========================================================
 ========================================================

for i in "${classifier_baselines[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs learn_classifiers --classifier_baselines $i --show_plots False --override_config_file $2 &
  pids[${k}]=$!
  k=$((k + 1))
done

 ========================================================
 ========================================================
 ========================================================

for i in "${gan_single_envs[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
  pids[${k}]=$!
  k=$((k + 1))
done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

#
#k=0
#pids=()
#
#for i in "${sources1[@]}"; do
#  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
#  pids[${k}]=$!
#  k=$((k + 1))
#done
#
## wait for all pids
#for pid in ${pids[*]}; do
#  wait $pid
#done
#
#
#
#
#k=0
#pids=()
#
#for i in "${sources2[@]}"; do
#  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
#  pids[${k}]=$!
#  k=$((k + 1))
#done
#
## wait for all pids
#for pid in ${pids[*]}; do
#  wait $pid
#done
#
#k=0
#pids=()
#
#for i in "${sources3[@]}"; do
#  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
#  pids[${k}]=$!
#  k=$((k + 1))
#done
#
## wait for all pids
#for pid in ${pids[*]}; do
#  wait $pid
#done
#
#k=0
#pids=()
#for i in "${targets1[@]}"; do
#  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
#  pids[${k}]=$!
#  k=$((k + 1))
#done
#
## wait for all pids
#for pid in ${pids[*]}; do
#  wait $pid
#done
#
#k=0
#pids=()
#for i in "${targets2[@]}"; do
#  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines "one_gan_by_env" --gan_single_envs $i --show_plots False --override_config_file "$2" &
#  pids[${k}]=$!
#  k=$((k + 1))
#done
#
## wait for all pids
#for pid in ${pids[*]}; do
#  wait $pid
#done


echo "models for one_gan_by_env done"

# ========================================================
# ========================================================
# ========================================================

k=0
pids=()

for i in "${gan_single_envs[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs inferences --show_plots False --envs_to_test "$i" --override_config_file "$2" --gan_baselines "one_gan_by_env" &
  pids[${k}]=$!
  k=$((k + 1))
done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

echo "tests  for one_gan_by_env done"

## ========================================================
## ========================================================
## ========================================================
#
k=0
pids=()

gan_baselines=(
  "hyper_gan_z_10"
  "hyper_gan_z_1"
  "agglo_gan"
  "hyper_nn"
  "hyper_nn_dropout"
  "hyper_nn_training_True"
  "hyper_gan_z_1_training_True"
  "hyper_gan_z_10_training_True"
  "hyper_gan_z_1_dropout"
  "hyper_gan_z_10_dropout"
)



for i in "${gan_baselines[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs learn_gans --gan_baselines $i --show_plots False --override_config_file $2 &
  pids[${k}]=$!
  k=$((k + 1))
done

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

echo "models for hyper/super gans done"

# ========================================================
# ========================================================
# ========================================================

k=0
pids=()
for i in "${gan_baselines[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs inferences --show_plots False --override_config_file $2 --gan_baselines "$i" &
  pids[${k}]=$!
  k=$((k + 1))
done
# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

echo "tests done"


# ========================================================
# ========================================================
# ========================================================

PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs print_test_models_results --show_plots False --override_config_file $2

echo "plots done"

#
#PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs cross_comparaison --show_plots False --override_config_file "$2"
#PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs test_models --show_plots False --override_config_file "$2"
#
#pids=()
#
#transfer_baselines=(
#  "solution_3"
#  "dqn"
#  "dyna_model_learning"
#)
#
#for i in "${transfer_baselines[@]}"; do
#  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs transfer --transfer_baselines $i --show_plots False --override_config_file "$2" &
#  pids[${k}]=$!
#done
#
## wait for all pids
#for pid in ${pids[*]}; do
#  wait $pid
#done
#
#echo "transfer done"
#
#PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs print_transfer_results --show_plots False --override_config_file "$2"
