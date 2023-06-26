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
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
import pytorch_lightning as pl
#from datasets import load_from_disk
import time
import matplotlib.pyplot as plt
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
#¡ This peft is used when there is no inputs_embeds
import sys
sys.path.insert(1, '/home/xl544/rds/hpc-work/Retrieval-Augmented-Visual-Question-Answering/src/models')
from PeftModel4Seq2SeqLM import PeftModelForSeq2SeqLM
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
        self.processor = Blip2Processor.from_pretrained(self.config.model_config.ModelVersion)
        self.generator = Blip2ForConditionalGeneration.from_pretrained(self.config.model_config.ModelVersion, torch_dtype=self.float_type)
        print('Uses full checkpoint from', self.config.model_config.ModelVersion)
        '''
        self.r=8
        print('r value for LoRA is', self.r)
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.r, lora_alpha=32, lora_dropout=0.1)
        self.generator = PeftModelForSeq2SeqLM(self.generator, peft_config)
        self.generator.print_trainable_parameters()
        '''

    def forward(self, questions,
                      gold_answer,
                      pixel_values,
                    **kwargs):
        
        #batch_size = input_ids.shape[0]
        #print('1',questions, gold_answer)
        batch_images_preprocessed = torch.stack(pixel_values).to(device)
        question = []
        for q in questions:
            question.append('Question: '+q+' Answer:')

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
        #generated_ids = self.generator.generate(**inputs)
        #img=batch_images_preprocessed[0].to('cpu')
        #plt.imshow(img.permute(1,2,0))

        #print('3',generated_ids.shape, generated_ids)
        #generated_text = self.processor.batch_decode(generated_ids)#, skip_special_tokens=True)
        #print(generated_text)
        return generator_outputs.loss


    def generate(self, questions,
                      pixel_values,
                      **kwargs):
        batch_images_preprocessed = torch.stack(pixel_values).to(device)
        question = []
        for q in questions:
            question.append('Question: '+q+' Answer:')
        inputs = self.processor(
            images=batch_images_preprocessed, 
            text=question, 
            padding='longest',
            max_length=self.config.data_loader.additional.max_decoder_source_length,
            truncation=True,
            return_tensors="pt").to(device, self.float_type)
        #print('2',inputs) #pixel_values,input_ids,attention_mask
        generated_ids = self.generator.generate(**inputs)
        #print('3',generated_ids.shape, generated_ids)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_text