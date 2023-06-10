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

import sys
sys.path.insert(1, '/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/src/models/blip2')
from blip2 import Blip2Base, disabled_train

data_dir = Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data")
okvqa_data_dir = data_dir / "ok-vqa"
vqa_data_dir = Path("")

def encode_img(device, base, image, ln_vision, visual_encoder, query_tokens, Qformer):
    #if self.low_resource:
    #    self.vit_to_cpu()
    #    image = image.to("cpu")
    with base.maybe_autocast(device=device): 
        image_embeds = ln_vision(visual_encoder(image)).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1).to(device)
        query_output = Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        #inputs_llama = self.llama_proj(query_output.last_hidden_state)
        #atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
    return query_output.last_hidden_state

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def main(vit_model_name: str, subtype: str = "val2014"):

    print(f"Extracting {subtype} using {vit_model_name} and qformer")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using ", device)
    #model, image_preprocessor = clip.load(clip_model_name, device=device)
    #clip_model_name = clip_model_name.replace('/', '_')
    
    img_size=224
    drop_path_rate=0
    use_grad_checkpoint=False
    vit_precision=""
    num_query_token=32
    q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"

    print('Loading VIT')
    base = Blip2Base()
    base.load_from_pretrained(url_or_filename=q_former_model)

    visual_encoder, ln_vision = base.init_vision_encoder(
        vit_model_name, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
    )
    #freeze vit not sure if needed
    for name, param in visual_encoder.named_parameters():
        param.requires_grad = False
    visual_encoder = visual_encoder.eval()
    visual_encoder.train = disabled_train
    for name, param in ln_vision.named_parameters():
        param.requires_grad = False
    ln_vision = ln_vision.eval()
    ln_vision.train = disabled_train

    visual_encoder.to(device)
    ln_vision.to(device)

    print('Loading Q-Former')
    Qformer, query_tokens = base.init_Qformer(
        num_query_token, visual_encoder.num_features
    )
    Qformer.cls = None
    Qformer.bert.embeddings.word_embeddings = None
    Qformer.bert.embeddings.position_embeddings = None
    for layer in Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None
    
    Qformer.to(device)
    query_tokens.to(device)

    out_path = (
        Path("/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa")
        / "pre-extracted_features"
        / "blip2_head_embeddings"
        / f"coco_{vit_model_name}_qformer_{subtype}.pkl"
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
    image_preprocessor = _transform(img_size)

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        img_id = str(data_item["image_id"])

        if img_id in img_ids_with_embeddings:
            continue

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        image = io.imread(img_path)
        image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)

        with torch.no_grad():
            #prefix = encode_image(image).cpu().numpy().astype(np.float32)
            prefix = encode_img(device, base, image, ln_vision, visual_encoder, query_tokens, Qformer).cpu().numpy().astype(np.float32)
            #print(prefix.shape) #(1,32,768)
        img_ids_with_embeddings[img_id] = prefix

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
        "--vit_model_type",
        default="eva_clip_g",
    )
    parser.add_argument(
        "--split", default="train2014", choices=("train2014", "val2014")
    )
    args = parser.parse_args()
    main(args.vit_model_type, args.split)
