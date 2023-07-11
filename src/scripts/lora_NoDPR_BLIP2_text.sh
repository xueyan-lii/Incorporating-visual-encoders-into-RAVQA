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
EXP_NAME=NoDPR-instructBLIP2-lora
# UNCOMMENT BELOW FOR SLURM SBATCH
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/11.1 intel/mkl/2017.4

JOBID=$SLURM_JOB_ID
LOG=../logs/$EXP_NAME-log.$JOBID
ERR=../logs/$EXP_NAME-err.$JOBID

python main.py ../configs/okvqa/T5_NoDPR_blip2_prefix+text.jsonnet \
    --mode train \
    --experiment_name ${EXP_NAME}.$JOBID  \
    --accelerator auto --devices 1  \
    --num_sanity_val_steps 2 \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=0.5  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.MLP_lr=0.0001 \
            train.scheduler=linear \
            model_config.UseInstructBLIP=1 \
   >> $LOG 2> $ERR