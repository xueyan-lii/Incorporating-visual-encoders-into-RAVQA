python main.py ../configs/okvqa/T5_NoDPR_prefix_only.jsonnet \
    --mode train \
    --experiment_name OKVQA_RA-VQA-NoDPR-debugging  \
    --accelerator auto --devices 1  \
    --num_sanity_val_steps 1 \
    --opts train.epochs=8  \
            train.batch_size=1  \
            valid.step_size=0.5  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.MLP_lr=0.0005 \
            train.scheduler=linear \
            model_config.UsePrefixEmb=0 \
            model_config.UseQformerEmb=0 \