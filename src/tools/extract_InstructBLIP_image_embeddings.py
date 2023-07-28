#difference from extract_instruct_head is that this one gets output from mlp
# to be used in faiss indexing
#can also do blip2, don't foget to change save file name
import torch
import pickle
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import os
from transformers import InstructBlipProcessor, Blip2Processor, Blip2ForConditionalGeneration
import sys
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import cv2 
sys.path.insert(1, '/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/src/models')
from instructblip_model import InstructBlipForConditionalGeneration

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform():
    return Compose([
        Resize((400,400), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        PILToTensor(),
    ])

def main(subtype: str = "val2014"):

    print(f"Extracting {subtype} using InstructBLIP Flan-T5-XL")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    #model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    
    model.to(device)
    
    out_path = (
        data_dir
        / "pre-extracted_features"
        / "text_embeddings"
        / f"BLIP2_image_embeddings_{subtype}.pkl"
    )

    with open(
        data_dir / f"OpenEnded_mscoco_{subtype}_questions.json", "r"
    ) as f:
        data = json.load(f)

    assert data["data_subtype"] == subtype, "wrong dataset subtype was loaded"
    print(data["info"])

    questions = data["questions"]
    print("%0d questions loaded from json " % len(questions))

    question_ids_with_embeddings = {}
    image_preprocessor = _transform()

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        question_id = str(data_item["question_id"])
        img_id = str(data_item["image_id"])

        if question_id in question_ids_with_embeddings:
            print("Already seen question id? Strange...")
            continue

        img_path = (
            data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        image = cv2.imread(img_path.as_posix())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)

        with torch.no_grad():
            question = str(data_item["question"])
            '''
            inputs = processor(
                images=image, 
                text=question, 
                return_tensors="pt").to(device, torch.float32)
            '''
            inputs = processor(
                images=image, 
                return_tensors="pt").to(device, torch.float32)
            language_model_inputs, language_model_attention_mask = model.get_qformer_features(**inputs)
            #print(language_model_inputs.shape) #[1, 32, 2048]
            text_embedding = language_model_inputs[0].cpu().numpy().astype(np.float32)
            #print(len(text_embedding)) #32

        question_ids_with_embeddings[question_id] =language_model_inputs[0].cpu().numpy().astype(np.float32)

        if (i + 1) % 10000 == 0:
            if not os.path.exists(out_path.parent):
                os.makedirs(out_path.parent)
            with open(out_path, "wb") as f:
                pickle.dump(question_ids_with_embeddings, f)
    if not os.path.exists(out_path.parent):
        os.makedirs(out_path.parent)
    with open(out_path, "wb") as f:
        pickle.dump(question_ids_with_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(question_ids_with_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="train2014", choices=("train2014", "val2014")
    )
    args = parser.parse_args()
    main(args.split)
