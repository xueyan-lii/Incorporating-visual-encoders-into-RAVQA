#extract question+oscar caption embedding together, max is 64 length so ratio to img is the same
#dont need gpu
import torch
import pickle
import json
from tqdm import tqdm
import argparse
from pathlib import Path
import clip
import numpy as np
import os
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import pickle

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
#vqa2_data_dir = data_dir / "vqa2"
def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        load_pickle_data = pickle.load(f)["cache"]
    data_dict = {
        str(item['question_id']):item
        for item in load_pickle_data['data_items']
    }
    return data_dict 

def main(subtype: str = "val2014"):

    print(f"Extracting {subtype} using InstructBLIP Flan-T5-XL")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    model.to(device)
    # load results from previous tests only for oscar/blip2 caption
    if subtype == "train2014":
        f = open('/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-train-K50.json')
    else:
        f = open('/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/testing_results/RAVQA-InstructBLIP-test-K50.json')
    captions = json.load(f)

    out_path = (
        data_dir
        / "pre-extracted_features"
        / "text_embeddings"
        / f"InstructBLIP_question+oscar_caption_embeddings_{subtype}.pkl"
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

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        question_id = str(data_item["question_id"])
        
        if question_id in question_ids_with_embeddings:
            print("Already seen question id? Strange...")
            continue
        
        #tokenized_question = clip.tokenize(data_item['question']).to(device)

        with torch.no_grad():
            print(data_item['question']+' '+captions[question_id]['oscar_caption'])
            inputs = processor( #output includes input_ids, attention_mask, qformer_input_ids,qformer_attention_mask, pixel_values
                text=data_item['question']+' '+captions[question_id]['oscar_caption'], 
                padding='max_length',
                max_length=64,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)
            text_embedding = model.get_input_embeddings()(inputs.input_ids)[0]
            print(text_embedding.shape) #torch.Size([32, 2048])
            text_embedding = model.get_input_embeddings()(inputs.input_ids)[0].cpu().numpy().astype(np.float32)
            #print(text_embedding)

        question_ids_with_embeddings[question_id] = text_embedding

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
