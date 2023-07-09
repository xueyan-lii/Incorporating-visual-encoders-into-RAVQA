#bf16 is enabled
python main.py ../configs/okvqa/RAVQA_instructblip.jsonnet  \
    --mode train \
    --experiment_name FrDPR_debugging \
    --accelerator auto --devices 1  \
    --modules freeze_question_encoder force_existence  \
    --precision bf16 \
    --opts train.epochs=8  \
            train.batch_size=1  \
            valid.step_size=0.5  \
            valid.batch_size=4  \
            train.additional.gradient_accumulation_steps=32  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.MLP_lr=0.0005 \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5 \