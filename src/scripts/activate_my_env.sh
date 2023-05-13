# source this file to activate the MLMI8.1 environment
# run these steps to create environment
module purge
module load miniconda3-4.5.4-gcc-5.4.0-hivczbz
module load slurm
eval "$(conda shell.bash hook)"
conda activate /.conda/envs/RAVQA
export TRANSFORMERS_CACHE="/home/$USER/rds/hpc-work/cache"
export HF_DATASETS_CACHE="/home/$USER/rds/hpc-work/cache"
export TOKENIZER_CACHE="/home/$USER/rds/hpc-work/cache"

export WDIR="/rds/user/$USER/hpc-work/Retrieval-Augmented-Visual-Question-Answering"

wandb login
