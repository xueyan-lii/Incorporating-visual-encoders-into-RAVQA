python main.py ../configs/okvqa/T5_NoDPR_prefix_only.jsonnet \
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
            model_config.PretrainedMLPPath=/home/xl544/rds/hpc-work/MLMI8_2022_VQA/Experiments/successful_runs/Pretrain_qformer_mlp_con_cap_T5-XL.22975003/train/saved_model/model_09.ckpt \
            model_config.UsePrefixEmb=0.5 \
            model_config.UseQformerEmb=1 \
            model_config.LoadPretrainedMLP=1 \
            model_config.TokenizerModelVersion="google/flan-t5-xl" \
            model_config.ModelVersion="google/flan-t5-xl" \