#!/bin/bash

#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_debug.yaml python "$HOME"/work/rltl 1 0
#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_swap.yaml python "$HOME"/work/rltl 1 0

function f(){
  for i in $(seq 1 $1); do echo $2; done
}

limit=10000

k=0
pids=()

#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_contextual.yaml python "$HOME"/work/rltl 3 1 &
f $limit "a" &
pids[${k}]=$!
k=$((k + 1))


#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_contextual_stochastic.yaml python "$HOME"/work/rltl 3 0 &
f $limit "b" &
pids[${k}]=$!
k=$((k + 1))

#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_generalisation.yaml python "$HOME"/work/rltl 7 7 &
f $limit "c" &
pids[${k}]=$!
k=$((k + 1))

#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_generalisation_stochastic.yaml python "$HOME"/work/rltl 6 5 &
f $limit "d" &
pids[${k}]=$!
k=$((k + 1))

# wait for all pids
for pid in ${pids[*]}; do
  wait $pid
done

f 10 "final"
#bash main.sh configs/nicolas/main.yaml configs/nicolas/main_final.yaml python "$HOME"/work/rltl 13 5

#sudo poweroff
