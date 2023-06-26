python main.py ../configs/okvqa/T5_NoDPR_blip2.jsonnet \
    --mode train \
    --experiment_name blip2_test  \
    --accelerator auto --devices 1  \
    --num_sanity_val_steps 5 \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=0.5  \
            valid.batch_size=2  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.MLP_lr=0.0003 \
            train.scheduler=linear \
   