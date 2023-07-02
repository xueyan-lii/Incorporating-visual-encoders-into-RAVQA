python main.py ../configs/okvqa/T5_NoDPR_prefix+text.jsonnet \
    --mode train \
    --experiment_name NoDPR_debug  \
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
            model_config.PretrainedMLPPath=/home/xl544/rds/hpc-work/MLMI8_2022_VQA/Experiments/successful_runs/Pretrain_Concap_vit_mlp1_t5large.22190303/train/saved_model/model_09.ckpt \
            model_config.UsePrefixEmb=1 \
            model_config.UseQformerEmb=0 \
            model_config.LoadPretrainedMLP=1 \
            model_config.TokenizerModelVersion="google/flan-t5-large" \
            model_config.ModelVersion="google/flan-t5-large" \