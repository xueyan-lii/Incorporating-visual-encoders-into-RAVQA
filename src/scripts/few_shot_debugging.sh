python main_fewshot.py ../configs/okvqa/few_shot.jsonnet  \
    --mode test \
    --experiment_name few_shot_1 \
    --accelerator auto --devices 1  \
    --log_prediction_tables   \
    --opts valid.batch_size=8 \
    data_loader.additional.num_shots=5 \
    data_loader.additional.score=1 \
    data_loader.additional.max_candidates='max' \
    data_loader.additional.caption_type="oscar" \