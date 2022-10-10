import torch
from torch import nn
import numpy as np

from encoder import Encoder
from decoder import Decoder
from layers import PositionalEncoding

from coco_completion import CocoCaptions


class Transformer(nn.Module):
	def __init__(self, word_to_idx, idx_to_word,
	             model_dim, fc_dim,
	             num_encoder_layer,
	             num_decoder_layer,
	             num_heads=4, dropout=0.1, device='cpu'):
		super(Transformer, self).__init__()

		self.word_to_idx = word_to_idx
		self.idx_to_word = idx_to_word

		self.null_idx = self.word_to_idx['<NULL>']
		self.start_idx = self.word_to_idx['<START>']
		self.end_idx = self.word_to_idx['<END>']

		self.word_embedding = nn.Embedding(num_embeddings=len(word_to_idx), embedding_dim=model_dim)
		self.positional_embedding = PositionalEncoding(model_dim, device=device)
		# self.positional_embedding = PositionalEncoding2(model_dim, dropout, device=device)

		self.encoder = Encoder(model_dim, num_encoder_layer, num_heads, fc_dim, dropout, device)
		self.decoder = Decoder(model_dim, num_decoder_layer, num_heads, fc_dim, dropout, device)
		self.prediction = nn.Linear(model_dim, len(word_to_idx))
		self.softmax = nn.Softmax(dim=-1)

		self.device = device
		self.to(device)

	def forward(self, X, Y):
		"""
		:param x: torch.Tensor, [batch_size, T1]
		:param y: torch.Tensor, [batch_size, T2]
		:return:
		"""
		src_mask = (X != self.null_idx).int()
		X = self.word_embedding(X) # [batch_size, T1, model_dim]
		X = self.positional_embedding(X)
		memory = self.encoder(X, src_mask)

		tgt_mask = (Y != self.null_idx).int()
		Y = self.word_embedding(Y)
		Y = self.positional_embedding(Y)

		out = self.decoder(memory, Y, src_mask, tgt_mask) # [batch_size, T2, model_dim]
		return self.prediction(out)

	def inference(self, x, y_start=None, max_length=100):
		"""
		:param x: torch.Tensor, [T1]
		:param y_start: torch.Tensor, [T2]
		:param max_length: int
		:return:
		"""
		self.eval()
		word_idx = [i for i in x.cpu().detach().numpy()]
		x = x[None, :]
		src_mask = (x != self.null_idx).int()
		x = self.word_embedding(x)  # [1, T1, model_dim]
		x = self.positional_embedding(x)  # [1, T1, model_dim]
		memory = self.encoder(x, src_mask)  # [1, T1, model_dim]

		if y_start is not None:
			y = y_start[None, :]
			word_idx.extend([
				i for i in y_start.cpu().detach().numpy()
			])
		else:
			y = torch.ones(1, 1).long().to(self.device) * self.start_idx
			word_idx.append(
				self.start_idx
			)

		while True:
			_, T = y.shape
			tgt_mask = torch.ones(1, T)
			target = self.word_embedding(y)
			target = self.positional_embedding(target)
			target = self.decoder(memory, target, src_mask, tgt_mask)
			distrib = self.prediction(target)

			idx = torch.argmax(distrib[0, -1, :], dim=-1)
			# distrib = self.softmax(distrib[0, -1, :]).cpu().detach().numpy()
			# idx = np.random.choice(np.arange(len(distrib)), p=distrib)
			word_idx.append(idx)

			if idx == self.end_idx or len(word_idx) >= max_length:
				return word_idx

			y = torch.cat([
				y, torch.ones(1, 1).long().to(self.device) * idx
			], dim=1)

	def translate(self, word_idx):
		return [
			self.idx_to_word[i] for i in word_idx
		]



def transformer_test():
	batch_size = 1
	T1 = 4
	T2 = 5
	dim = 16
	ds = CocoCaptions(val=True)

	X = torch.randint(0, 5, [batch_size, T1])
	Y = torch.randint(0, 5, [batch_size, T2])
	model = Transformer(
		word_to_idx=ds.word_to_idx,
		idx_to_word=ds.idx_to_word,
		model_dim=dim,
		fc_dim=dim, num_encoder_layer=2, num_decoder_layer=2, num_heads=1
	)
	model(X, Y)

# transformer_test()