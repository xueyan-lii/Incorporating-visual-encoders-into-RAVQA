from PIL import Image
import requests
from torchvision.transforms import Compose, Resize, PILToTensor, ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode

from PIL import Image
import cv2 

def _convert_image_to_rgb(image):
    return image.convert("RGB")
def _transform():
    return Compose([
        Resize((400,400), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        PILToTensor(),
        ])
def _transform1():
    return Compose([
        Resize((250,250), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        ])
image_preprocessor = _transform()
image_preprocessor1 = _transform1()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
img = image_preprocessor(image) 
plt.imshow(img.permute(1,2,0))
plt.savefig('onlinepic.jpg')

path_list=["/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/pig.png",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000042.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000073.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000074.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000133.jpg",
"/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/data/ok-vqa/debug_pics/COCO_val2014_000000000136.jpg",
]
count=0
for img_path in path_list:
    print(count)
    img = cv2.imread(img_path)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print(img)
    img = image_preprocessor(Image.fromarray(img)) 
    #print(img)
    img = image_preprocessor1(Image.fromarray(img[0].numpy()))  
    #img = image_preprocessor1(img)  
    #plt.imshow(img)
    plt.imshow(img.permute(1,2,0))
    
    plt.savefig(str(count)+'.jpg')
    count+=1
    

