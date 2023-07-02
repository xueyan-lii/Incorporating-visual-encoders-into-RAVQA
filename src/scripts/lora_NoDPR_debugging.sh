python main.py ../configs/okvqa/T5_NoDPR_prefix_only_lora.jsonnet \
    --mode train \
    --experiment_name lora_test  \
    --accelerator auto --devices 1  \
    --num_sanity_val_steps 1 \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=0.5  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.MLP_lr=0.0001 \
            train.scheduler=linear \
            model_config.UsePrefixEmb=0 \
            model_config.UseQformerEmb=0 \
            model_config.LoadPretrainedMLP=0 \
            model_config.TokenizerModelVersion="google/flan-t5-xl" \
            model_config.ModelVersion="google/flan-t5-xl" \
   