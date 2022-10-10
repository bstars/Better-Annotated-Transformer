import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from transformer import Transformer
from coco_completion import CocoCaptions
from config import Params

def overfit(model:Transformer,
           training_set:CocoCaptions,
           val_set:CocoCaptions, batch_size, lr, epochs, device):

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss(reduce=False)
	model.to(device)
	loader = DataLoader(training_set, batch_size, shuffle=True, collate_fn=training_set.collate_fn)

	def loss_mask(y):
		mask = torch.ones_like(y).to(device)
		for word, weight in Params.words_weight.items():
			idx = torch.where( y == training_set.word_to_idx[word] )
			mask[idx] = weight
		return mask

	for e in range(epochs):
		for i, (x,y) in enumerate(loader):

			x, y = x.to(device), y.to(device)
			for _ in range(1000):
				optimizer.zero_grad()

				pred = model(x, y[:,:-1]) # [batch_size, seq_length, vocab_size]
				loss = criterion(pred.transpose(1,2), y[:,1:])
				# print(y.shape, loss.shape)
				mask = loss_mask(y[:,1:])
				loss = torch.mean(loss * mask)

				loss.backward()
				optimizer.step()
				print(loss.item())
				idx = np.random.randint(0, batch_size)
				print(
					model.translate(model.inference(x[idx]))
				)

def train(model:Transformer,
           training_set:CocoCaptions,
           val_set:CocoCaptions, batch_size, lr, epochs, device):

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss(reduce=False)
	model.to(device)
	loader = DataLoader(training_set, batch_size, shuffle=True, collate_fn=training_set.collate_fn)

	def loss_mask(y):
		mask = torch.ones_like(y)
		for word, weight in Params.words_weight.items():
			idx = torch.where( y == training_set.word_to_idx[word] )
			# print(idx)
			mask[idx] = weight
		return mask

	for e in range(epochs):
		for i, (x,y) in enumerate(loader):

			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()

			pred = model(x, y[:,:-1]) # [batch_size, seq_length, vocab_size]
			loss = criterion(pred.transpose(1,2), y[:,1:])
			mask = loss_mask(y[:,1:])
			loss = torch.mean(loss * mask)

			loss.backward()
			optimizer.step()
			# print(loss.item())
			# print(
			# 	model.translate(model.inference(x[0]))
			# )

			if i % 20 == 0:
				print( '%d epochs, %d/%d, loss=%.5f' % (e, i, len(loader), loss.item()) )

				idx = np.random.randint(0, len(val_set))
				x_, y_ = val_set[idx]
				x_ = x_.to(device)
				print(
					model.translate(model.inference(x_))
				)
		if e % 10 == 0:
			torch.save(
				{ 'model_state_dict' : model.state_dict() },
				os.path.join(Params.ckpt_saving_path, "%d.pth" % (e))
			)

if __name__ == '__main__':
	training_set = CocoCaptions(val=False)
	val_set = CocoCaptions(val=True)
	model = Transformer(
		training_set.word_to_idx,
		training_set.idx_to_word,
		model_dim=128,
		fc_dim=128,
		num_encoder_layer=3,
		num_decoder_layer=3,
		num_heads=4,
		device=Params.device)

	train(model,
	      training_set,
	      val_set,
	      Params.batch_size,
	      Params.learning_rate,
	      epochs=Params.epochs,
	      device=Params.device
	)

	# overfit(model,
	#        training_set,
	#        val_set,
	#        batch_size=128,
	#        lr=1e-3,
	#        epochs=Params.epochs,
	#        device=Params.device
	#        )