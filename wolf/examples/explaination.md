### How to run the code

## training

observation: veh_speed, rel_speed, distance_headway, follow_rel_speed, follow_distance_headway, speed_limit, prev_accel

action: acceleration

single agent: sabcm_ddpg.py

multi-agent: mabcm_ddpg.py

## testing

BCM: bcm_ddpg_test.py

CFM: car_follow_ddpg_test.py

QEW: carfollowing_sumo_qew.py

potential issue: 

1. dimension issue (change DDPG dimension)
2. key error issue (?)
3. network issue (car_following_env, sumo_qew_new)



## save model

checkpoint: tmp/ xxx
tensorboard: runs/ xxx 

