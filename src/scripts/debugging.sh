python main.py \
    ../configs/okvqa/RAVQA.jsonnet  \
    --mode train  \
    --experiment_name OKVQA_RA-VQA-FrDPR_debugging \
    --accelerator auto --devices 1  \
    --modules freeze_question_encoder force_existence  \
    --log_prediction_tables \
    --num_sanity_val_steps 2 \
    --opts train.epochs=10  \
            train.batch_size=2  \
            valid.step_size=1  \
            valid.batch_size=32  \
            train.additional.gradient_accumulation_steps=16  \
            train.lr=0.00006  \
            train.retriever_lr=0.00001  \
            train.MLP_lr=0.0005 \
            train.scheduler=linear  \
            data_loader.additional.num_knowledge_passages=5 \
            model_config.UsePrefixEmb=1 \
            model_config.UseQformerEmb=1 \
           # model_config.DecoderTokenizerModelVersion="google/flan-t5-large" \
           # model_config.GeneratorModelVersion="google/flan-t5-large" \