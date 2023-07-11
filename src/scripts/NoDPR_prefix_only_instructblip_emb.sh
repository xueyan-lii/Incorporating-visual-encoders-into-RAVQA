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
EXP_NAME=OKVQA_NoDPR-flanT5large-instruct
# UNCOMMENT BELOW FOR SLURM SBATCH
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/11.1 intel/mkl/2017.4
#source scripts/activate_my_env.sh
JOBID=$SLURM_JOB_ID
LOG=../logs/$EXP_NAME-log.$JOBID
ERR=../logs/$EXP_NAME-err.$JOBID

python main.py ../configs/okvqa/T5_NoDPR_prefix_only_instructblip_emb.jsonnet \
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
            model_config.UsePrefixEmb=0.5 \
            model_config.PretrainedMLPPath=/home/xl544/rds/hpc-work/MLMI8_2022_VQA/Experiments/successful_runs/Pretrain_instructblip_mlp_con_cap.23317389/train/saved_model/model_09.ckpt \
            model_config.LoadPretrainedMLP=1 \
            model_config.TokenizerModelVersion="google/flan-t5-large" \
            model_config.ModelVersion="google/flan-t5-large" \
   >> $LOG 2> $ERR
# train.load_model_path="../Experiments/Pretrain_qformer_mlp_con_cap.22181357/train/saved_model/model_04.ckpt" \
 #           model_config.UsePrefixEmb=0.5 \
 #           model_config.UseQformerEmb=1 \
# once UsePrefixEmb is set to 0, no prefix is used (only question exist, MLP not trained), setting UseQformerEmb will not do anything
# set UseQformerEmb=1 to use vit emb
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