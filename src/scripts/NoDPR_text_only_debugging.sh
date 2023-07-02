python main.py ../configs/okvqa/baseline_T5.jsonnet \
    --mode train \
    --experiment_name OKVQA_RA-VQA-NoDPR  \
    --accelerator auto --devices 1  \
    --num_sanity_val_steps 2 \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=0.5  \
            valid.batch_size=2  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.MLP_lr=0.0001 \
            train.scheduler=linear \
            model_config.UsePrefixEmb=0.5 \
            model_config.UseQformerEmb=0 \
            model_config.TokenizerModelVersion="google/flan-t5-large" \
            model_config.ModelVersion="google/flan-t5-large" \