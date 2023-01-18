#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=2-00:00:00
#SBATCH --output=gagru-input-full-20210327-pems0408.out
#SBATCH --account=def-ssanner

module load python/3.7
source ~/prediction/bin/activate


python ~/tune_ga_gru.py --path /home/bruceli/data/pems/PEMS04/PEMS04.npz --csv /home/bruceli/data/pems/PEMS04/PEMS04.csv --dataname pems04 --att input
python ~/tune_ga_gru.py --path /home/bruceli/data/pems/PEMS08/PEMS08.npz --csv /home/bruceli/data/pems/PEMS04/PEMS08.csv --dataname pems08 --att input
python ~/tune_ga_gru.py --path /home/bruceli/data/pems/PEMS04/PEMS04.npz --csv /home/bruceli/data/pems/PEMS04/PEMS04.csv --dataname pems04 --att full
python ~/tune_ga_gru.py --path /home/bruceli/data/pems/PEMS08/PEMS08.npz --csv /home/bruceli/data/pems/PEMS04/PEMS08.csv --dataname pems08 --att full