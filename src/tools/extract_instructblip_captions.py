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
import json
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import cv2 
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
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
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model.to(device)
    print('Uses full checkpoint from Salesforce/instructblip-flan-t5-xl')
    
    out_path = (
        Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
        / "pre-extracted_features"
        / "blip2_head_embeddings"
        / f"coco_instructblip_captions_type1_{subtype}.json"
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
        question=question+" Provide a caption for the image that will help answer the question​"
        #print(question)
        inputs = processor(
            images=image, 
            text=question, 
            return_tensors="pt").to(device, torch.float32)
        #generator_outputs = model.generate(**inputs)
        #generated_text = processor.batch_decode(generator_outputs, skip_special_tokens=True)
        #print('caption', generated_text)
        #img_ids_with_embeddings[str(data_item["question_id"])] = generated_text[0]
        
        #full_question="A photo of" #type1, not good, captions are just short answers
        #full_question=" Taking the previous question into account, describe this image" #type2
        #full_question = "Provide a caption for the image that will help answer the question"
        full_question = question
        print(full_question)
        with torch.no_grad():
            full_question_tokens = processor(
                text=full_question, 
                return_tensors="pt").to(device, torch.float32)

            generator_outputs = model.generate(
                pixel_values=inputs.pixel_values,
                qformer_input_ids=inputs.qformer_input_ids,
                qformer_attention_mask=inputs.qformer_attention_mask,
                input_ids=full_question_tokens.input_ids,
                attention_mask=full_question_tokens.attention_mask)

            generated_text = processor.batch_decode(generator_outputs, skip_special_tokens=True)
            print('caption', generated_text)
            img_ids_with_embeddings[str(data_item["question_id"])] = generated_text
        


    if not os.path.exists(out_path.parent):
        os.makedirs(out_path.parent)
    json_object = json.dumps(img_ids_with_embeddings, indent=4)
    with open(out_path, "wb") as f:
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
