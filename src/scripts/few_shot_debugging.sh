#to count tokens, model_config.ModelClass=FewShotModelGPT 
#check that prompt is correct in model
python main_fewshot.py ../configs/okvqa/few_shot.jsonnet  \
    --mode test \
    --experiment_name few_shot_promptcap_altprompt \
    --accelerator auto --devices 1  \
    --log_prediction_tables   \
    --precision bf16 \
    --opts valid.batch_size=4 \
    model_config.ModelClass=FewShotModel \
    data_loader.additional.num_shots=5 \
    data_loader.additional.score=1 \
    data_loader.additional.max_candidates=5 \
    data_loader.additional.caption_type="promptcap" \