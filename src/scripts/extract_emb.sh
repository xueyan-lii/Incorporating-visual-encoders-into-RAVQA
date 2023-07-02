#!/bin/bash
#SBATCH -J RAVQA_FrDPR_train_cc
#SBATCH -A MLMI-xl544-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#SBATCH -p ampere
#! ############################################################

LOG=/dev/stdout
ERR=/dev/stderr
EXP_NAME=extract_hfaceQformer_embedding
# UNCOMMENT BELOW FOR SLURM SBATCH
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/11.1 intel/mkl/2017.4
#source scripts/activate_my_env.sh
JOBID=$SLURM_JOB_ID
LOG=../logs/$EXP_NAME-log.$JOBID
ERR=../logs/$EXP_NAME-err.$JOBID

python tools/extract_VitQFormer_huggingface_version.py --split train2014 \
   >> $LOG 2> $ERR
