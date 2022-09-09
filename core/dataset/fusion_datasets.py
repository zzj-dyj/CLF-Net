import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch

class Fusion_Datasets(Dataset):
	"""docstring for Fusion_Datasets"""

	def __init__(self, configs, transform=None, is_Test=False):
		super(Fusion_Datasets, self).__init__()
		self.root_dir = configs['root_dir']
		self.transform = transform
		self.channels = configs['channels']
		self.sensors = configs['sensors']
		self.img_list = {i: os.listdir(os.path.join(self.root_dir, i)) for i in self.sensors}
		self.img_path = {i: [os.path.join(self.root_dir, i, j) for j in os.listdir(os.path.join(self.root_dir, i))]
		                 for i in self.sensors}
		self.is_Test = is_Test

	def __getitem__(self, index):
		img_data_train = {}
		img_data_test = {}
		for i in self.sensors:
			img = input_setup(self.img_path[i][index])
			if self.transform is not None:
				img = self.transform(img)
			img = img.type(torch.cuda.FloatTensor)
			img_data_train.update({i: img})
			img_data_test.update({i: img})

		if self.is_Test:
			return img_data_test
		else:
			return img_data_train

	def __len__(self):
		img_num = [len(self.img_list[i]) for i in self.img_list]
		img_counter = Counter(img_num)
		assert len(img_counter) == 1, 'Sensors Has Different length'
		return img_num[0]


def input_setup(data_path):
	_ir = imread(data_path)
	input_ir = (_ir - 127.5) / 127.5
	# input_ir = _ir / 255
	return input_ir


def imread(path):
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	return img[:, :, 0]



