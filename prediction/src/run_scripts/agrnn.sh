#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=1-00:00:00
#SBATCH --output=agrnn20210319-pems08.out
#SBATCH --account=def-ssanner

module load python/3.7
source ~/prediction/bin/activate

python ~/tune_agrnn.py
