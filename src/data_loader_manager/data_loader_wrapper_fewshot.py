import sys
import time
import json
import copy
import numpy as np
import json
import torch

from tqdm import tqdm
from copy import deepcopy
from easydict import EasyDict

import logging
logger = logging.getLogger(__name__)


class DataLoaderWrapperFewShot():
    '''
    Data loader wrapper, general class definitions
    '''

    def __init__(self, config):
        self.config = config

    def build_dataset(self):
        """
        This function loads data and features required for building the dataset
        """

        self.data = EasyDict()

        dataset_modules = self.config.data_loader.dataset_modules.module_list
        for dataset_module in dataset_modules:
            module_config = self.config.data_loader.dataset_modules.module_dict[dataset_module]
            logger.info('Loading dataset module: {}'.format(module_config))
            loading_func = getattr(self, dataset_module)
            loading_func(module_config)
            print('data columns: {}'.format(self.data.keys()))
            
