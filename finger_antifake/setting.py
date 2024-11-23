import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.functional import F

class Envirment:
    def __init__(self,_type):
        self._type = _type # ['train','test']
        self.device = 'cpu' if torch.cuda.is_available else 'cuda'
        self.batch_size = 0
        self.init_envirment()
        
    def init_envirment(self):
        if self._type == "train" or self._type == 'TRAIN':
            self.batch_size = 64
        elif self._type == 'test' or self._type == 'TEST':
            self.batch_size = 1

train_envirment = Envirment('train')
test_envirment = Envirment("test")