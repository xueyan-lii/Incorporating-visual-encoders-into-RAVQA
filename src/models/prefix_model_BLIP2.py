import copy
import math
import os
from turtle import forward
import warnings
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from collections import Counter, defaultdict
from easydict import EasyDict
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as T
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
import pytorch_lightning as pl

#from datasets import load_from_disk
import time
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForSeq2SeqLM
#¡ This peft is used when there is no inputs_embeds
import sys
sys.path.insert(1, '/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/src/models')
#from PeftModel4Seq2SeqLM import PeftModelForSeq2SeqLM
from InstructBLIP_lora import InstructBLIPLora
#used when github instructblip is used
#sys.path.insert(1, '/home/xl544/rds/hpc-work/LAVIS')
#from lavis.models import load_model_and_preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrefixModelBLIP2(pl.LightningModule):
    '''
    Class to use full blip2 model
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config
        #self.data_loader = data_loader
        #self.tokenizer = data_loader.tokenizer
        if device == "cpu":
            self.float_type = torch.float32
        else:
            self.float_type = torch.float16

        if not self.config.model_config.UseInstructBLIP:
        
            self.processor = Blip2Processor.from_pretrained(self.config.model_config.ModelVersion)
            self.generator = Blip2ForConditionalGeneration.from_pretrained(self.config.model_config.ModelVersion, torch_dtype=self.float_type)
            print('Uses full checkpoint from', self.config.model_config.ModelVersion)
            
            
        else:
            '''
            self.transform = T.ToPILImage()
            self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)
            
            '''
            self.generator = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            #self.generator = InstructBLIPLora.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
            print('Uses full checkpoint from Salesforce/instructblip-flan-t5-xl')
            
            self.r=8
            print('r value for LoRA is', self.r)
            peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.r, lora_alpha=32, lora_dropout=0.1)
            self.generator.language_model = PeftModelForSeq2SeqLM(self.generator.language_model, peft_config)
            self.generator.language_model.print_trainable_parameters()


    def forward(self, questions,
                      gold_answer,
                      pixel_values,
                    **kwargs):
        if not self.config.model_config.UseInstructBLIP: #use blip2 hgface package
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            question = []
            for q in questions:
                question.append('Use the provided image to answer the question: '+q+' Provide your answer as short as possible:')

            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=question, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, self.float_type)
            #print('2', inputs)

            labels = self.processor(
                text=gold_answer, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt")#.to(device, self.float_type)

            #print('3',labels)
            #print('4',labels.input_ids)
            generator_outputs = self.generator(
                pixel_values=inputs.pixel_values, 
                input_ids=inputs.input_ids, 
                attention_mask=inputs.attention_mask, 
                labels=labels.input_ids.to(device))
            #print(question, gold_answer, generator_outputs.loss)
            return generator_outputs.loss
            #generated_ids = self.generator.generate(**inputs)
            #generated_text = self.processor.batch_decode(generated_ids)#, skip_special_tokens=True)
            #print(generated_text)
            
        else:
            ''' 
            #use instructBLIP from github
            images=[]
            for i in range(len(pixel_values)):
                images.append(self.vis_processors["eval"](self.transform(pixel_values[i])))
            images = torch.stack(images).to(device)
            generator_outputs = self.model({"image": images, "text_input": questions, "text_output": gold_answer})
            #print('generator_outputs',generator_outputs)
            return generator_outputs['loss']
            '''
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=questions, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)#instructblip doesn't like float16 with gpu

            labels = self.processor(
                text=gold_answer, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt")

            generator_outputs = self.generator(
                **inputs, #includes input_ids, attention_mask, qformer_input_ids,qformer_attention_mask, pixel_values
                labels=labels.input_ids.to(device))
            #print(questions, gold_answer, generator_outputs.loss)
            return generator_outputs.loss


    def generate(self, questions,
                      pixel_values,
                      **kwargs):
        if not self.config.model_config.UseInstructBLIP:
        
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            question = []
            for q in questions: 
                question.append('Use the provided image to answer the question: '+q+' Provide your answer as short as possible:')
                
            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=question, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, self.float_type)
            #print('2',inputs) #pixel_values,input_ids,attention_mask
            generated_ids = self.generator.generate(**inputs)
            #print('3',generated_ids.shape) #2,12
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        else:
            '''
            images=[]
            for i in range(len(pixel_values)):
                images.append(self.vis_processors["eval"](self.transform(pixel_values[i]).to(device)))
            images = torch.stack(images).to(device)
            generated_text = self.model.generate({"image": images, "prompt": questions})
            '''
            batch_images_preprocessed = torch.stack(pixel_values).to(device)
            inputs = self.processor(
                images=batch_images_preprocessed, 
                text=questions, 
                padding='longest',
                max_length=self.config.data_loader.additional.max_decoder_source_length,
                truncation=True,
                return_tensors="pt").to(device, torch.float32)

            generator_outputs = self.generator.generate(**inputs)
            generated_text = self.processor.batch_decode(generator_outputs, skip_special_tokens=True)
            
        return generated_text