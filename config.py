import torch

class Params:
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# data parameters
	coco_path = './'

	# training parameter
	learning_rate = 1e-3
	batch_size = 16
	epochs = 50

	#
	ckpt_saving_path = './'


	#
	words_weight = {
		"<NULL>": 0.,
		"<UNK>": 0.5,
		"with": 0.95,
		'a': 0.95
	}

