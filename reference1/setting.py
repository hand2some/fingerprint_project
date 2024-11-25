"""
The file is All project setting
I provide two mode of environment:
(1) train
(2) test
In sometime,i need test my model through just one data,so i set the test data size on 1

"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.functional import F


class Environment:
	def __init__(self, _type):
		self._type = _type  # ['train','test']
		self.device = 'cpu' if torch.cuda.is_available else 'cuda'
		self.batch_size = 1  # batch size
		self.lr = 1e-3  # learning rate
		self.init_environment()

	def init_environment(self):
		if self._type == "train" or self._type == 'TRAIN':
			if self.device == "cpu":
				self.batch_size = 4
			else:
				self.batch_size = 64
		elif self._type == 'test' or self._type == 'TEST':
			self.batch_size = 1


train_environment = Environment('train')
test_environment = Environment("test")
