import os, json
import numpy as np
import h5py
import urllib.request, urllib.parse, os, tempfile
from urllib.error import URLError, HTTPError
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import Params

def load_coco():
	path = Params.coco_path
	data = {}
	caption_file = os.path.join(path, 'coco2014_captions.h5')

	with h5py.File(caption_file, 'r') as f:
		for k, v in f.items():
			data[k] = np.asarray(v)

	dict_file = os.path.join(path, 'coco2014_vocab.json')
	with open(dict_file, 'r') as f:
		dict_data = json.load(f)
		for k, v in dict_data.items():
			data[k] = v

	return data

class CocoCaptions(Dataset):
	def __init__(self, val=False):
		super(CocoCaptions, self).__init__()
		data_dict = load_coco()

		self.captions = data_dict['train_captions'] if not val else data_dict['val_captions']
		self.idx_to_word = data_dict['idx_to_word']
		self.word_to_idx = data_dict['word_to_idx']

		self.null_idx = self.word_to_idx['<NULL>']
		self.start_idx = self.word_to_idx['<START>']
		self.end_idx = self.word_to_idx['<END>']


	def __len__(self):
		return len(self.captions)

	def __getitem__(self, idx):
		sentence = self.captions[idx, 1:]
		end_idx = np.where(sentence==self.end_idx)[0][0]
		sentence = sentence[:end_idx].tolist()

		split_idx = int(len(sentence) * 0.4)

		x = sentence[:split_idx]
		y = [self.start_idx] + sentence[split_idx:] + [self.end_idx]

		return torch.Tensor(x).long(), torch.Tensor(y).long()

	def collate_fn(self, batchs):
		xs = []
		ys = []
		for x, y in batchs:
			xs.append(x)
			ys.append(y)

		return pad_sequence(xs, batch_first=True, padding_value=self.null_idx), \
		       pad_sequence(ys, batch_first=True, padding_value=self.null_idx)