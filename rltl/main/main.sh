#!/bin/bash

gan_single_envs=()
gan_baselines=(
  "hyper_gan_z_5"
  "hyper_gan_z_10"
  "one_gan_by_env_z_10"
  "agglo_gan"
  "hyper_nn"
  "hyper_nn_training_True"
)

classifier_baselines=(
  "feedforward"
  "full_bayesian"
)

one_gan_by_env="one_gan_by_env_z_10"

sources=()
targets=()
for i in $(seq 0 $5); do
  gan_single_envs+=("Gridworld_sources-$i")
  sources+=("Gridworld_sources-$i")
done

for i in $(seq 0 $6); do
  gan_single_envs+=("Gridworld_targets-$i")
  targets+=("Gridworld_targets-$i")
done

# ========================================================
# ========================================================
# ========================================================

createdatasets() {
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
}

# ========================================================
# ========================================================
# ========================================================

learnclassifier() {

  k=0
  pids=()

  for i in "${classifier_baselines[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs learn_classifiers --classifier_baselines $i --show_plots False --override_config_file $2 &
    pids[${k}]=$!
    k=$((k + 1))
  done

  # wait for all pids
  for pid in ${pids[*]}; do
    wait $pid
  done
}

# ========================================================
# ========================================================
# ========================================================

oneganbyenv() {
  k=0
  pids=()

  for i in "${gan_single_envs[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines $one_gan_by_env --gan_single_envs $i --show_plots False --override_config_file "$2" &
    pids[${k}]=$!
    k=$((k + 1))
  done

  # wait for all pids
  for pid in ${pids[*]}; do
    wait $pid
  done

  echo "models for one_gan_by_env done"
}

oneganbyenvallandclassifier() {
  k=0
  pids=()

  for i in "${gan_single_envs[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs learn_gans --gan_baselines $one_gan_by_env --gan_single_envs $i --show_plots False --override_config_file "$2" &
    pids[${k}]=$!
    k=$((k + 1))
  done

  for i in "${classifier_baselines[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs learn_classifiers --classifier_baselines $i --show_plots False --override_config_file $2 &
    pids[${k}]=$!
    k=$((k + 1))
  done

  # wait for all pids
  for pid in ${pids[*]}; do
    wait $pid
  done

  echo "models for classifier and one_gan_by_env done"
}
# ========================================================
# ========================================================
# ========================================================
inferencesoneganbyenv() {
  k=0
  pids=()

  for i in "${gan_single_envs[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs "inferences" --show_plots False --envs_to_test "$i" --override_config_file "$2" --gan_baselines "one_gan_by_env" &
    pids[${k}]=$!
    k=$((k + 1))
  done

  # wait for all pids
  for pid in ${pids[*]}; do
    wait $pid
  done

  echo "tests  for one_gan_by_env done"
}
## ========================================================
## ========================================================
## ========================================================
#
learngans() {

  k=0
  pids=()

  for i in "${gan_baselines[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs "learn_gans" --gan_baselines "$i" --show_plots False --override_config_file "$2" --envs_to_test "all" &
    pids[${k}]=$!
    k=$((k + 1))
  done

  # wait for all pids
  for pid in ${pids[*]}; do
    wait $pid
  done

  echo "models for hyper/super gans done"
}

# ========================================================
# ========================================================
# ========================================================
inferencegans() {

  k=0
  pids=()
  for i in "${gan_baselines[@]}"; do
    PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --envs_to_test "all" --jobs "inferences" --show_plots False --override_config_file "$2" --gan_baselines "$i" &
    pids[${k}]=$!
    k=$((k + 1))
  done
  # wait for all pids
  for pid in ${pids[*]}; do
    wait $pid
  done

  echo "tests done"
}

# ========================================================
# ========================================================
# ========================================================

plot() {

  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --jobs print_test_models_results --show_plots False --override_config_file "$2" --gan_baselines "all" --envs_to_test "all"

  echo "plots done"
}

testclassifier() {
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --override_config_file "$2" --show_plots False --jobs "cross_comparaison_classifier" --envs_to_test "all" &
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py "$1" --override_config_file "$2" --show_plots False --jobs "visualize_classifier" --envs_to_test "all" &
}

#createdatasets $1 $2 $3 $4 $5

#oneganbyenvallandclassifier $1 $2 $3 $4 $5

# or do this

#oneganbyenv $1 $2 $3 $4 $5
#learnclassifier $1 $2 $3 $4 $5


#testclassifier $1 $2 $3 $4 $5
#inferencesoneganbyenv $1 $2 $3 $4 $5
learngans $1 $2 $3 $4 $5
#inferencegans $1 $2 $3 $4 $5
#plot $1 $2 $3 $4 $5



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
