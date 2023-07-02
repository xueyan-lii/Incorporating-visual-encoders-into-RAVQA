from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torchvision.transforms as T
import torch
from torchvision.transforms import Compose, ToTensor, Resize, PILToTensor
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
import sys
sys.path.insert(1, '/home/xl544/rds/hpc-work/LAVIS')
from lavis.models import load_model_and_preprocess

from PIL import Image
import cv2 

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    float_type = torch.float32
else:
    float_type = torch.float16

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform():
    return Compose([
        Resize((400,400), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        PILToTensor(),
        ])
image_preprocessor = _transform()
transform = T.ToPILImage()

# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)

path_list=["/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000042.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000073.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000074.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000133.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000136.jpg"]
count=11
for img_path in path_list:
    count+=1
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = image_preprocessor(Image.fromarray(img))  

    image = vis_processors["eval"](transform(img)).unsqueeze(0).to(device)
    answer = model.generate({"image": image, "prompt": "What is unusual about this image?"})
    print(answer)
