Incorporating visual-encoders into retrieval-augmented visual question answering.

This continuation of the original RAVQA (Weizhe Lin and Bill Byrne 2022) codebase adds modern vision‑language models and few-shot prompting pipeline for closed source models.

New in this fork
- InstructBLIP generator with efficient Q‑Former reuse and LM conditioning
- RAG with InstructBLIP (dynamic/static retrieval, RAG‑sequence marginalization, pseudo‑labels)
- BLIP‑2 and InstructBLIP baselines (full model, text‑based‑vision fusion)
- T5 prefix tuning with LoRA for parameter‑efficient finetuning
- Few‑shot retrieval utilities and scripts
- Prophet prompting pipeline and OK‑VQA evaluation helpers

Quick start (scripts)
```bash
bash src/scripts/RAVQA_instructblip.sh
```

No‑DPR baselines
```bash
bash src/scripts/NoDPR_prefix+text.sh
bash src/scripts/NoDPR_text_only.sh
bash src/scripts/NoDPR_prefix_only.sh
```

LoRA + BLIP‑2 variants
```bash
bash src/scripts/lora_NoDPR_BLIP2.sh
bash src/scripts/lora_NoDPR_BLIP2_text.sh
```

Few‑shot
```bash
bash src/scripts/few_shot.sh
```

Prophet pipeline (prompting + evaluation)
```bash
bash prophet-main/scripts/prompt.sh
bash prophet-main/scripts/evaluate_file.sh
```

InstructBLIP and RAG highlights
- `src/models/instructblip_model.py`: separates vision/Q‑Former from the T5 LM; exposes
  - `get_qformer_features(...)` to precompute image‑conditioned embeddings once
  - `forward(...)`/`generate(...)` that accept `inputs_embeds` + text for multi‑doc decoding
- `src/models/rag/rag_model_instructblip.py` + `src/trainers/rag_executor_instructblip.py`:
  - Dynamic retrieval via HF/FAISS indices or static DPR tops
  - RAG‑sequence marginalization combining doc priors with token logprobs
  - Pseudo‑label variants (`Approach5/6/7/8`, `NoPR`) and `force_existence`
  - Freezing knobs: `freeze_question_encoder`, `freeze_generator`

Text‑based‑vision and BLIP‑2
- Full BLIP‑2/InstructBLIP wrappers: `src/models/prefix_model_BLIP2.py`, `src/models/prefix_model_BLIP2_text.py`, `src/models/blip2/*`
- Text fusion: VinVL + OCR → text context concatenated with the question for InstructBLIP

Prefix tuning with LoRA (T5)
- `src/models/prefix_model_lora.py`: MLP projects image/Q‑Former features into prefix embeddings, prepended to T5 token embeddings; LoRA on LM

Data loaders and datasets (OK‑VQA)
- Image and text‑fusion pipelines: `src/data_loader_manager/data_loader_blip2.py`, `src/data_loader_manager/data_loader_blip2_text.py`, `src/data_loader_manager/datasets/okvqa_datasets_BLIP2.py`
- Caching for preprocessed data, VinVL/OCR features, and embeddings

Configs (examples)
- InstructBLIP + RAG: `configs/okvqa/RAVQA_instructblip.jsonnet`
- BLIP‑2 NoDPR: `configs/okvqa/T5_NoDPR_blip2*.jsonnet`
- Prefix/LoRA variants: `configs/okvqa/T5_NoDPR_prefix_only*.jsonnet`
- Few‑shot: `configs/okvqa/few_shot.jsonnet`


Prophet and evaluation
- Prompt configs and task mapping under `prophet-main/configs/*`
- OK‑VQA evaluation utilities under `prophet-main/evaluation/*`
