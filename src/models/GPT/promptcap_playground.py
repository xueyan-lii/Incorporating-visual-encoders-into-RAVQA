import torch
from promptcap import PromptCap

model = PromptCap("vqascore/promptcap-coco-vqa")  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"

if torch.cuda.is_available():
  model.cuda()

prompt = "please describe this image according to the given question: where is the animal?"
image = "/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000042.jpg"

print(model.caption(prompt, image))