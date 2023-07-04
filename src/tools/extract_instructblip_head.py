# !! requires higher transfromer version, use BLIP2 conda environment
# embedding is question dependent so question id is stored with each img rather than img id
import torch
import skimage.io as io
from PIL import Image
import pickle
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import os
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import cv2 
from transformers import InstructBlipProcessor
import sys
sys.path.insert(1, '/home/xl544/rds/hpc-work/MLMI8_2022_VQA/MLMI-VQA-2022/src/models')
from instructblip_model import InstructBlipModel

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data")
okvqa_data_dir = data_dir / "ok-vqa"
vqa_data_dir = Path("")
model_dir = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data"

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform():
    return Compose([
        Resize((400,400), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        PILToTensor(),
    ])

def main(subtype: str = "val2014"):

    print(f"Extracting {subtype} using BLIP2 qformer output")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using ", device)

    print('Loading InstructBLIP')
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model = InstructBlipModel.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model.to(device)
    print('Uses full checkpoint from Salesforce/instructblip-flan-t5-xl')
    
    out_path = (
        Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
        / "pre-extracted_features"
        / "blip2_head_embeddings"
        / f"coco_instructblip_qformer_{subtype}.pkl"
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
    image_preprocessor = _transform()

    for i in tqdm(range(len(questions))):
        data_item = questions[i]
        print(data_item)

        img_id = str(data_item["image_id"])

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        #image = io.imread(img_path)
        image = cv2.imread(img_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        question = str(data_item["question"])
        inputs = processor(
            images=image, 
            text=question, 
            return_tensors="pt").to(device, torch.float32)
        with torch.no_grad():
            prefix = model.get_qformer_features(**inputs)
            print('prefix shape',prefix.shape) #[1, 32, 768]
        
        img_ids_with_embeddings[str(data_item["question_id"])] = prefix.to(torch.float32)

        if (i + 1) % 10000 == 0:
            if not os.path.exists(out_path.parent):
                os.makedirs(out_path.parent)
            with open(out_path, "wb") as f:
                pickle.dump(img_ids_with_embeddings, f)

    if not os.path.exists(out_path.parent):
        os.makedirs(out_path.parent)
    with open(out_path, "wb") as f:
        pickle.dump(img_ids_with_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(img_ids_with_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="train2014", choices=("train2014", "val2014")
    )
    args = parser.parse_args()
    main(args.split)
