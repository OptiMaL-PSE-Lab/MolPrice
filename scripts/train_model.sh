#!/bin/bash
#PBS -j oe
#PBS -lselect=1:ncpus=12:mem=450gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=09:00:00

export WANDB_INIT_TIMEOUT=300
export WANDB_HTTP_TIMEOUT=300
export WANDB_API_KEY="your_wandb_api_key"


module load anaconda3/personal
cd $PBS_O_WORKDIR
cd ..
source activate graphfg

python -m bin.train --model $1 --fp $2