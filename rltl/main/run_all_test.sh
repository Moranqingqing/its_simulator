#!/bin/bash

#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_debug.yaml python "$HOME"/work/rltl 1 0
#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_swap.yaml python "$HOME"/work/rltl 1 0

function main_contextual(){
  bash main.sh configs/nicolas/main.yaml configs/nicolas/main_contextual.yaml python "$HOME"/work/rltl 3 1
}

function main_contextual_stochastic(){
  bash main.sh configs/nicolas/main.yaml configs/nicolas/main_contextual_stochastic.yaml python "$HOME"/work/rltl 3 0

}

function main_generalisation(){
  bash main.sh configs/nicolas/main.yaml configs/nicolas/main_generalisation.yaml python "$HOME"/work/rltl 7 7
}

function main_generalisation_stochastic(){
  bash main.sh configs/nicolas/main.yaml configs/nicolas/main_generalisation_stochastic.yaml python "$HOME"/work/rltl 6 5
}

function main_final(){
  bash main.sh configs/nicolas/main.yaml configs/nicolas/main_final.yaml python "$HOME"/work/rltl 13 5
}

k=0
pids=()

main_contextual &
pids[${k}]=$!
k=$((k + 1))


main_contextual_stochastic &
pids[${k}]=$!
k=$((k + 1))

main_generalisation &
pids[${k}]=$!
k=$((k + 1))

main_generalisation_stochastic &
pids[${k}]=$!
k=$((k + 1))

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

main_final

sudo poweroff
