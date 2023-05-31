python main.py \
    ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA-FrDPR_testrun \
    --accelerator auto --devices 1  \
    --modules freeze_question_encoder force_existence  \
    --log_prediction_tables \
    --num_sanity_val_steps 0 \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5 \