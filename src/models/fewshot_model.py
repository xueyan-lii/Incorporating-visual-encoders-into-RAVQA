# few shot learning with various llms
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

#from datasets import load_from_disk
import time
import matplotlib.pyplot as plt
#from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

class FewShotModel(pl.LightningModule):
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
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
        '''
        self.tokenizer = LlamaTokenizer.from_pretrained("/home/xl544/rds/rds-cvnlp-hirYTW1FQIw/shared_space/converted_vicuna_weights/vicuna-13b")
        self.model = LlamaForCausalLM.from_pretrained("/home/xl544/rds/rds-cvnlp-hirYTW1FQIw/shared_space/converted_vicuna_weights/vicuna-13b")
        '''

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
    
    def create_shot_no_candidates(self, caption, question, candidates, scores, max_candidates, answer=''):
        context = ' Context: ' + caption
        question = ' Question: ' + question
        answer = ' Answer: ' + answer
        return context + question + answer

    def combine_shots(self, prompt_head, shots, test_shot):
        return prompt_head + ''.join(shots) + test_shot

    def generate(self, questions, gold_answers, in_context_examples):
        #print(questions,gold_answers)
        #print(in_context_examples)
        max_candidates = self.config.data_loader.additional.max_candidates
        caption_type = self.config.data_loader.additional.caption_type + "_caption"
        
        prompt_head = 'Answer the question according to the context and answer candidates. Each answer candidate is associated with a confidence score within a bracket. The true answer may not be included in the candidates.'  
        #prompt_head = 'Answer the question according to the context and answer candidates. Each answer candidate is associated with a confidence score within a bracket. Come up with an answer if none of the answer candidates are suitable.'
        #prompt_head = 'Answer the question according to the context and answer candidates. Each answer candidate is associated with a confidence score within a bracket. Choose one answer from the candidates.'
        
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
                shot=self.combine_shots(prompt_head, shots, test_shot)
                combined_shots.append(shot)
            '''
            print('\n',sample['question_id'])
            print(sample['gold_answer'])
            print(sample['answers'])
            print(self.combine_shots(prompt_head, shots, test_shot))
            '''
        #print(combined_shots)
        
        inputs = self.tokenizer(combined_shots, padding="longest", return_tensors="pt").to(device) 
        outputs = self.model.generate(**inputs)
        generated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #remove any brackets or scores
        
        generated_text_postprocessed=[]
        for answer in generated_text:
            bracket_index = answer.find('(')
            if bracket_index == -1:
                generated_text_postprocessed.append(answer.strip())
            else:
                generated_text_postprocessed.append(answer[:bracket_index].strip())
        
        '''
        #this section is for llama2
        generated_text=[]
        for prompt in combined_shots:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device) 
            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=10)
            answer = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #print(answer)
            #since this model repeat the whole prompt, only get last answer
            start=[m.start() for m in re.finditer('Answer:', answer)][-1]
            end=[m.start() for m in re.finditer('Context', answer)][-1]
            print(answer[start:])
            if end > start:
                generation=answer[start+7:end].strip()
            elif '\n' in answer:
                loc=[m.start() for m in re.finditer('\n', answer)][0]
                if loc > start:
                    generation=answer[start+7:loc].strip()
            else:
                generation=answer[start+7:].strip()
            generation = generation.replace(".", "")
            generated_text.append(generation)
        '''
        print(generated_text)
        print(generated_text_postprocessed)
        return generated_text_postprocessed, combined_shots