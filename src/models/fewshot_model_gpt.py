# counting number of tokens for gpt3.5, number stored in predictions
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
#from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import re
import pytorch_lightning as pl
import tiktoken
#from datasets import load_from_disk
import time
import matplotlib.pyplot as plt
#from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

class FewShotModelGPT(pl.LightningModule):
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

        #load LLM
        

    def forward(self, **kwargs):
       pass

    def create_shot(self, caption, question, candidates, scores, max_candidates, answer=''):
        context = ' Context: ' + caption
        question = ' Question: ' + question
        if max_candidates == 'm':
            candidates = ' Candidates: ' + ''.join([i+' ('+j+'), ' for i,j in zip(candidates, scores)])
        else:
            candidates = ' Candidates: ' + ''.join([i+' ('+j+'), ' for i,j in zip(candidates[:max_candidates], scores[:max_candidates])])
        #candidates = ' Candidates: ' + ''.join([i+', ' for i,j in zip(candidates[:max_candidates], scores[:max_candidates])]) #for no score

        answer = ' Answer: ' + answer
        return context + question + candidates + answer

    def combine_shots(self, prompt_head, shots, test_shot):
        return prompt_head + ''.join(shots) + test_shot
    
    def num_tokens_from_string(self, string):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def generate(self, questions, gold_answers, in_context_examples):
        #print(questions,gold_answers)
        #print(in_context_examples)
        max_candidates = self.config.data_loader.additional.max_candidates
        caption_type = self.config.data_loader.additional.caption_type + "_caption"
        
        prompt_head = 'Answer the question according to the context and answer candidates. Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates'
        #prompt_head = ''
        combined_shots = []
        num_tokens = []
        
        for sample in in_context_examples:
            #the same val information is in each shot so only take the first one
            
            test_shot = self.create_shot(caption=sample['in_context_examples'][0]['val_'+caption_type], 
                                    question=sample['in_context_examples'][0]['val_question'],
                                    candidates=sample['in_context_examples'][0]['val_doc_predictions'],
                                    scores=sample['in_context_examples'][0]['val_doc_scores'], 
                                    max_candidates=max_candidates)
            
            if self.config.data_loader.additional.num_shots == 0:
                combined_shots.append(self.combine_shots('', '', test_shot))
                #combined_shots.append(self.combine_shots(prompt_head, '', test_shot))
            else:
                shots=[]
                for shot in sample['in_context_examples']:
                    shots.append(self.create_shot(shot[caption_type], shot['question'], shot['doc_predictions'], shot['doc_scores'], max_candidates, shot['gold_answer']))
                one_sample=self.combine_shots(prompt_head, shots, test_shot)
                length = self.num_tokens_from_string(one_sample)
                if length>4000:
                    print(sample['in_context_examples'][0]['val_question'],'has token length exceed 4000',length)
                num_tokens.append(length+5) #assume answers are 5 length
                combined_shots.append(one_sample)
            '''
            print('\n',sample['question_id'])
            print(sample['gold_answer'])
            print(sample['answers'])
            print(self.combine_shots(prompt_head, shots, test_shot))
            '''
        #print(combined_shots)
        
        #inputs = self.tokenizer(combined_shots, padding="longest", return_tensors="pt").to(device) 
        #outputs = self.model.generate(**inputs)
        #generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(num_tokens)
        return num_tokens, combined_shots