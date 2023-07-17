# testing only from checkpoint
python main.py \
    ../configs/okvqa/RAVQA_instructblip.jsonnet  \
    --mode test  \
    --experiment_name InstructBLIP_RAVQA_test \
    --accelerator auto --devices 1  \
    --modules force_existence  \
    --precision bf16 \
    --log_prediction_tables   \
    --opts data_loader.additional.num_knowledge_passages=5 \
            test.load_model_path=/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/Experiments/OKVQA-instructblip.23647945/train/saved_model/model_3.ckpt \