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
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, T5PreTrainedModel
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForSeq2SeqLM
import time
#import sys
#sys.path.insert(1, '/home/xl544/rds/hpc-work/MLMI8_2022_VQA/MLMI-VQA-2022/src/models')
#from vct0_qformer import VCT0Model_Qformer, VCT0Prefix_Qformer
#from vct0 import VCT0Model, VCT0Prefix

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class PrefixModel(pl.LightningModule):
    '''
    Class to only add image prefix embedding to question without any text captioning or document
    '''
    def __init__(self, config: EasyDict, data_loader) -> None:
        super().__init__()

        self.config = config
        self.data_loader = data_loader
        self.tokenizer = data_loader.tokenizer

        # Initialising generator
        GeneratorModelClass = globals()[self.config.model_config.GeneratorModelClass]
        GeneratorConfigClass = globals()[self.config.model_config.ConfigClass]
        generator_model_config = GeneratorConfigClass.from_pretrained(self.config.model_config.ModelVersion)
        self.generator = GeneratorModelClass.from_pretrained(self.config.model_config.ModelVersion,config=generator_model_config)
        '''
        self.r=8
        print('r value for LoRA is', self.r)
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=self.r, lora_alpha=32, lora_dropout=0.1)
        self.generator = PeftModelForSeq2SeqLM(self.generator, peft_config)
        self.generator.print_trainable_parameters()
        '''                                          
        self.generator.resize_token_embeddings(len(self.tokenizer))
        self.lm_embedding_size = self.generator.model_dim 
        self.prefix_length = 32
        self.prefix_size = 768  
            
        if self.config.model_config.UsePrefixEmb == 2:
            print("\n Using MLP")
            print("MLP input ",self.prefix_size, ' Hidden dim ',(self.lm_embedding_size * self.prefix_length) // 2, ' Output dim ',self.lm_embedding_size,' x ', self.prefix_length)
            self.clip_project = MLP(
                (
                    self.prefix_size,
                    (self.lm_embedding_size * self.prefix_length) // 2,
                    self.lm_embedding_size * self.prefix_length,
                )
            )
        elif self.config.model_config.UsePrefixEmb == 1:
            print("\n Using MLP")
            print("MLP input ",self.prefix_size,' Output dim ',self.lm_embedding_size,' x ', self.prefix_length)
            self.clip_project = MLP(
                (
                    self.prefix_size,
                    self.lm_embedding_size * self.prefix_length,
                )
            )

        #for qformer op, prefix_length has to be 32 cannot be changed
        elif self.config.model_config.UsePrefixEmb == 0.5: 
            print("\n Using MLP for Qformer")
            print("MLP input ",self.prefix_size,' Output dim ',self.lm_embedding_size)
            self.clip_project = MLP(
                (
                    self.prefix_size,
                    self.lm_embedding_size,
                )
            )

        else:
            print('\n No MLP or image prefix used')

        if self.config.model_config.LoadPretrainedMLP:
            checkpoint_path = Path(self.config.model_config.PretrainedMLPPath)
            print('Weights before loading')
            for n, p in self.clip_project.named_parameters():
                print(n,p)
            print('Loading MLP checkpoint from',checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            #print('1',checkpoint['state_dict'].keys())
            
            #checkpoint['state_dict']["clip_project.model.0.weight"] = checkpoint['state_dict'].pop('model.clip_project.model.0.weight')
            #checkpoint['state_dict']["clip_project.model.0.bias"] = checkpoint['state_dict'].pop("model.clip_project.model.0.bias")
            #self.VCT0 = VCT0Prefix(model_version=self.config.model_config.ModelVersion)
            #self.VCT0.load_state_dict(checkpoint['state_dict'], strict=False) 
            #self.clip_project.load_state_dict(self.VCT0.clip_project.state_dict())
            with torch.no_grad():
                self.clip_project.model[0].weight.copy_(checkpoint['state_dict']["model.clip_project.model.0.weight"])
                self.clip_project.model[0].bias.copy_(checkpoint['state_dict']["model.clip_project.model.0.bias"])
            print('Weights after loading')
            for n, p in self.clip_project.named_parameters():
                print(n,p)
        else:
            print('No pretrained MLP loaded')

    def insert_prefix_into_emb(self, batch_size, batch_text_tokens, batch_text_masks, batch_prefix_projections, labels):
        no_documents=1
        prefix_len = batch_prefix_projections.shape[1]
        text_len = batch_text_tokens.shape[1]
        emb_dim = batch_prefix_projections.shape[2]
        batch_text_embeddings = self.generator.shared(batch_text_tokens)

        embedding_out=torch.ones((no_documents*batch_size, prefix_len+text_len, emb_dim), dtype=int, device=labels.device)*-100.0 
        mask_out=torch.ones((no_documents*batch_size, prefix_len+text_len),dtype=int, device=labels.device)*-100
        batch_inds_for_broadcasting=[[i] for i in range(no_documents*batch_size)]
        text_tokens_mask=torch.zeros((no_documents*batch_size, prefix_len+text_len),dtype=int, device=labels.device)
        text_tokens_mask[batch_inds_for_broadcasting, torch.arange(text_len)+prefix_len] =1
        embedding_out[text_tokens_mask.bool()]=batch_text_embeddings.view(-1,emb_dim)
        for i in range(batch_size):
            prefix_mask = torch.zeros((no_documents*batch_size, prefix_len+text_len),dtype=int, device=labels.device).bool()
            prefix_mask[[[j] for j in range(i*no_documents, (i+1)*no_documents)], torch.arange(prefix_len, device=labels.device)]=1
            embedding_out[prefix_mask] = batch_prefix_projections[i].repeat(no_documents,1,1).view(-1,emb_dim)

        mask_out[~text_tokens_mask.bool()]=1
        mask_out[text_tokens_mask.bool()] = batch_text_masks.view(-1)
        return embedding_out, mask_out
        

    def forward(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      labels: torch.Tensor,
                      prefix: torch.Tensor,
                    **kwargs):
        
        batch_size = input_ids.shape[0]

        if self.config.model_config.UsePrefixEmb:
            prefix = prefix.to(device=input_ids.device, dtype=torch.float)
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.lm_embedding_size)
            joint_embeddings, joint_attention_masks = self.insert_prefix_into_emb(batch_size=batch_size, batch_text_tokens=input_ids, 
                                            batch_text_masks=attention_mask, 
                                            batch_prefix_projections=prefix_projections,
                                            labels=labels)
            
            generator_outputs = self.generator(
                            inputs_embeds=joint_embeddings,
                            attention_mask=joint_attention_masks,
                            labels=labels)
        
        else:
            generator_outputs = self.generator(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        
        return EasyDict(loss=generator_outputs.loss)


    def generate(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      prefix: torch.Tensor,
                      **kwargs):

        batch_size = input_ids.shape[0]
        
        if self.config.model_config.UsePrefixEmb:
            prefix = prefix.to(device=input_ids.device, dtype=torch.float)
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.lm_embedding_size)
            joint_embeddings, joint_attention_masks = self.insert_prefix_into_emb(batch_size=batch_size, batch_text_tokens=input_ids, 
                                            batch_text_masks=attention_mask, 
                                            batch_prefix_projections=prefix_projections,
                                            labels=input_ids)
            
            test_batch = EasyDict({
            'inputs_embeds': joint_embeddings,
            'attention_mask': joint_attention_masks,
            })
        
        else:
            test_batch = EasyDict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        })
        
        # Get encoder outputs first
        encoder_outputs = self.generator.encoder(
            **test_batch
        )

        # Get decoder outputs from encoder_outputs
        test_batch = {
            'encoder_outputs': encoder_outputs,
            "max_length": self.config.data_loader.additional.max_target_length,
        }
        generation_outputs = self.generator.generate(**test_batch)
        return generation_outputs