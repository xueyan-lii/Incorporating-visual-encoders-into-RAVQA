# !! requires higher transfromer version, use BLIP2 conda environment
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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import Blip2Processor, Blip2Model

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data")
okvqa_data_dir = data_dir / "ok-vqa"
vqa_data_dir = Path("")
model_dir = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data"

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform():
    return Compose([
        _convert_image_to_rgb,
        ToTensor(),
    ])

def main(subtype: str = "val2014"):

    print(f"Extracting {subtype} using raw pixel values")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using ", device)
    
    out_path = (
        Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
        / "pre-extracted_features"
        / "raw-pixels"
        / f"okvqa_raw_pixels_{subtype}.pkl"
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

        img_id = str(data_item["image_id"])

        if img_id in img_ids_with_embeddings:
            continue

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        image = io.imread(img_path)
        image = image_preprocessor(Image.fromarray(image)).to(device)
        #print(image.shape) #3,x,x not reshaped
        img_ids_with_embeddings[img_id] = image

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
