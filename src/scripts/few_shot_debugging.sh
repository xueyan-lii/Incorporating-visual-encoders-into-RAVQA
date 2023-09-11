#to count tokens, model_config.ModelClass=FewShotModelGPT 
#check that prompt is correct in model
python main_fewshot.py ../configs/okvqa/few_shot.jsonnet  \
    --mode test \
    --experiment_name few_shot_final_1 \
    --accelerator auto --devices 1  \
    --log_prediction_tables   \
    --precision bf16 \
    --opts valid.batch_size=2 \
    model_config.ModelClass=FewShotModel \
    data_loader.additional.num_shots=20 \
    data_loader.additional.answer_type=3 \
    data_loader.additional.max_candidates=15 \
    data_loader.additional.caption_type="promptcap" \