#!/bin/sh

ssh john "cd /home/ncarrara/work/rltl/rltl/main;python main.py configs/nicolas/main.yaml --override_config_file configs/nicolas/main_generalisation.yaml"


#ssh john "/home/ncarrara/anaconda3/bin/tensorboard --logdir=/home/ncarrara/work/rltl/rltl/main/data/main_generalisation/0/learn_classifiers/full_bayesian/bayesian_classifier/tf_log/dynamics"