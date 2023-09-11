#!/bin/bash
# This script is used to prompt GPT-3 to generate final answers.

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2;;
    --version)
      VERSION="$2"
      shift 2;;
    --examples_path)
      EXAMPLES_PATH="$2"
      shift 2;;
    --openai_key)
      OPENAI_KEY="$2"
      shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

TASK=${TASK:-ok} # task name, one of ['ok', 'aok_val', 'aok_test'], default 'ok'
VERSION=${VERSION:-"prompt_okvqa"} # version name, default 'prompt_for_$TASK'
EXAMPLES_PATH=${EXAMPLES_PATH:-"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/prophet-main/data/few_shot_final_1.csv"} # path to all prompts
OPENAI_KEY=${OPENAI_KEY:-"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"} 
# CHECK RESUME AFTER FIRST RUN!!!!!!!!!!
# use blip2 env
# CUDA_VISIBLE_DEVICES=$GPU \
python main.py \
    --task $TASK --run_mode prompt \
    --version $VERSION \
    --cfg configs/prompt.yml \
    --examples_path $EXAMPLES_PATH \
    --openai_key $OPENAI_KEY \
    --resume