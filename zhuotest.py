import torch
from tqdm import tqdm
import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from learning3d.losses import RMSEFeaturesLoss, FrobeniusNormLoss
from learning3d.models import PointNet, PointNetLK
from learning3d.data_utils import ModelNet40Data,RegistrationData

# create the checkpoints folders
def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
	os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')


class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()