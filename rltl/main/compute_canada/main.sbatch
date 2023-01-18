#!/bin/bash

#SBATCH --mail-user=nicolas.carrara1u@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module load python/3.7
module load cuda cudnn
source /home/ncarrara/projects/def-ssanner/ncarrara/ENV/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/ncarrara/projects/def-ssanner/ncarrara/work/rltl
pushd  /home/ncarrara/projects/def-ssanner/ncarrara/work/rltl/rltl/main
bash main.sh ${config_file} compute_canada/ncarrara_override.yaml
popd
~