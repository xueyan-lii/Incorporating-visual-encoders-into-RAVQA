#!/bin/bash
#SBATCH -J RAVQA_FrDPR_train_cc
#SBATCH -A MLMI-xl544-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue
#SBATCH -p ampere
#! ############################################################

LOG=/dev/stdout
ERR=/dev/stderr
EXP_NAME=OKVQA_RA-VQA-FrDPR_FullCorpus
# UNCOMMENT BELOW FOR SLURM SBATCH
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/11.1 intel/mkl/2017.4
#source scripts/activate_my_env.sh
JOBID=$SLURM_JOB_ID
LOG=../logs/$EXP_NAME-log.$JOBID
ERR=../logs/$EXP_NAME-err.$JOBID

python main.py \
    ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name ${EXP_NAME}.$JOBID \
    --accelerator auto --devices 1  \
    --modules freeze_question_encoder force_existence  \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5 \
   >> $LOG 2> $ERR

# testing only from checkpoint
# python main.py \
#    ../configs/okvqa/RAVQA.jsonnet  \
#    --mode test  \
#    --experiment_name ${EXP_NAME}.$JOBID \
#    --accelerator auto --devices 1  \
#    --modules freeze_question_encoder force_existence  \
#    --log_prediction_tables   \
#    --opts data_loader.additional.num_knowledge_passages=5 \
#            test.load_model_path=../Experiments/OKVQA_RA-VQA-FrDPR_FullCorpus.19792250/train/saved_model/model_6.ckpt \
#   >> $LOG 2> $ERR