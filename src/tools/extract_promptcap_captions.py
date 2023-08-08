#use minigpt4 environment
import torch
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import os
import json
from promptcap import PromptCap#import sys
#sys.path.insert(1, '/home/xl544/rds/hpc-work/MLMI8_2022_VQA/MLMI-VQA-2022/src/models')
#from instructblip_model import InstructBlipModel

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data")
okvqa_data_dir = data_dir / "ok-vqa"
vqa_data_dir = Path("")
model_dir = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data"


def main(subtype: str = "val2014"):

    print(f"Extracting {subtype} using promptcap to caption")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using ", device)

    model = PromptCap("vqascore/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"
    if torch.cuda.is_available():
        model.cuda()

    out_path = (
        Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
        / "pre-extracted_features"
        / "captions"
        / f"coco_promptcap_captions_{subtype}.json"
    )

    with open(
        okvqa_data_dir / f"OpenEnded_mscoco_{subtype}_questions.json", "r"
    ) as f:
        data = json.load(f)

    assert data["data_subtype"] == subtype, "wrong dataset subtype was loaded"
    print(data["info"])

    questions = data["questions"]
    print("%0d questions loaded from json " % len(questions))

    img_ids_with_embeddings = {}

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        img_id = str(data_item["image_id"])

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        question = str(data_item["question"])
        prompt = "please describe this image according to the given question: "+question
        print(prompt)
        generated_text = model.caption(prompt, img_path)
        print('generated_text',generated_text)
        img_ids_with_embeddings[str(data_item["question_id"])] = generated_text
        
        
    if not os.path.exists(out_path.parent):
        os.makedirs(out_path.parent)
    json_object = json.dumps(img_ids_with_embeddings, indent=4)
    with open(out_path, "w") as outfile:
        outfile.write(json_object)

    print("Done")
    print("%0d embeddings saved " % len(img_ids_with_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="train2014", choices=("train2014", "val2014")
    )
    args = parser.parse_args()
    main(args.split)
