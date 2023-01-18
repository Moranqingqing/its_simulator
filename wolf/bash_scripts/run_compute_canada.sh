#!/bin/bash
#SBATCH --account=rrg-ssanner
#SBATCH --gres=gpu:1           # Number of GPUs (per node)
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G              # memory (per node)
#SBATCH --time=0-48:00         # time (DD-HH:MM)

#SBATCH --mail-user=parthjaggi@iitrpr.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

# source ~/env/traffic4/bin/activate
source ~/.bashrc

env=$1
config=$2
override=$3

# Can run following example commands. To be run from the sow45_code directory.
# sbatch wolf/ray/compute_canada/run.sh test4_1 iql_global_reward_no_dueling.yaml wolf/tests/override_configs/parth.yaml
# sbatch wolf/ray/compute_canada/run.sh test4_1 iql_global_reward_no_dueling_progression.yaml wolf/tests/override_configs/parth.yaml
# sbatch wolf/ray/compute_canada/run.sh test4_1 iql_global_reward_noop.yaml wolf/tests/override_configs/parth.yaml
# sbatch wolf/ray/compute_canada/run.sh test4_1 iql_global_reward_phase_select.yaml wolf/tests/override_configs/parth.yaml

python wolf/ray/main.py wolf/tests/traffic_env/$env/$config $override
