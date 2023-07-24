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
import json
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import cv2 
from transformers import Blip2Processor, Blip2ForConditionalGeneration
#import sys
#sys.path.insert(1, '/home/xl544/rds/hpc-work/MLMI8_2022_VQA/MLMI-VQA-2022/src/models')
#from instructblip_model import InstructBlipModel

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

    print(f"Extracting {subtype} using BLIP2 to caption")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using ", device)

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b",torch_dtype=torch.float32)
    model.to(device)
    print('Uses full checkpoint from Salesforce/blip2-opt-2.7b')
    
    out_path = (
        Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
        / "pre-extracted_features"
        / "captions"
        / f"coco_blip2_captions_{subtype}.json"
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
            print('image already captioned')
            continue

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        #image = io.imread(img_path)
        image = cv2.imread(img_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        prompt = "A photo of"

        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt").to(device, torch.float32)

        with torch.no_grad():
            generator_outputs = model.generate(**inputs)
            generated_text = processor.batch_decode(generator_outputs, skip_special_tokens=True)[0].strip()
            print('generated_text',generated_text)
            img_ids_with_embeddings[img_id] = generated_text
        
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
