import os
import re
import sys
import time
import json
import copy
from tqdm import tqdm
import csv
import json
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import cv2
import base64

from copy import deepcopy
from pprint import pprint
from easydict import EasyDict
from collections import defaultdict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import logging

logger = logging.getLogger(__name__)

from utils.dirs import create_dirs
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from utils.cache_system import save_cached_data, load_cached_data
from torchvision.utils import make_grid, save_image

from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from data_loader_manager.module_parser import ModuleParser


class FewShotDataset(torch.utils.data.Dataset, ModuleParser):
    """
    Base VQA2 dataset class
    """

    def __init__(self, config, dataset_dict):
        logger.info(f"initialising {type(self).__name__}...")
        self.mode = dataset_dict["mode"]
        self.config = config
        self.data = dataset_dict["data"]
        self.in_context_examples = dataset_dict["in_context_examples"]
        self.answer_candidate_list = dataset_dict["answer_candidate_list"]

    def __len__(self):
        return len(self.data.data_items)

    def __getitem__(self, idx):
        item = self.data.data_items[idx]
        num_shots = self.config.data_loader.additional.num_shots
        in_context_examples = self.in_context_examples.get(str(item.question_id))
        if num_shots == 0:
            #need info in the contexts
            in_context_examples = in_context_examples[:1]
        else:
            in_context_examples = in_context_examples[:num_shots]
            
        
        sample = EasyDict(
            {
                "question_id": item.question_id,
                "question": item.question,
                "img_key_full": item.img_key_full,
                "gold_answer": item.gold_answer,
                "answers": item.answers,
                "in_context_examples": in_context_examples,
            }
        )
        return sample

    def collate_fn(self, batch):
        #############################
        #  Meta Features
        #############################
        question_ids = [sample.question_id for sample in batch]
        questions = [sample.question for sample in batch]
        answers = [sample.answers for sample in batch]
        gold_answers = [sample.gold_answer for sample in batch]

        in_context_img_keys = [[str(item['image_key']).zfill(12) for item in sample['in_context_examples']] for sample in batch]
        batched_data = EasyDict(
            {
                "question_ids": question_ids,
                "questions": questions,
                "answers": answers,
                "gold_answers": gold_answers,
                "in_context_img_keys": in_context_img_keys,
                "in_context_examples": batch,
            }
        )

        return batched_data
