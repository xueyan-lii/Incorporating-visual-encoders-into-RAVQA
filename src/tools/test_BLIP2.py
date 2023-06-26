from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from torchvision.transforms import Compose, ToTensor, Resize, PILToTensor
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode


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
        Resize((500,500), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        PILToTensor(),
        ])
image_preprocessor = _transform()

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=float_type
)
model.to(device)

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
    
    prompt = "Question: describe this picture? Answer:"
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device, float_type)
    imag = inputs.pixel_values
    plt.imshow(imag[0].float().cpu().permute(1,2,0))
    plt.savefig(str(count)+'.jpg')

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)
    prompt = "Question: what colors are in this picture? Answer:"
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device, float_type)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(generated_text)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "Question: describe this picture. Answer:"
image = image_preprocessor(image)  
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, float_type)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

prompt = "Question: what color is the background? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, float_type)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

imag = inputs.pixel_values
plt.imshow(imag[0].float().cpu().permute(1,2,0))
plt.savefig('cats.jpg')