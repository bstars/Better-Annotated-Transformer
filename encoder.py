import torch
from torch import nn
from layers import MultiHeadAttention, FeedForward

class EncoderLayer(nn.Module):
	def __init__(self, model_dim, num_heads=4, fc_dim=512, dropout=0.1, device='cpu'):
		super(EncoderLayer, self).__init__()
		self.attn = MultiHeadAttention(model_dim, model_dim, model_dim, model_dim, num_heads, device)
		self.norm1 = nn.LayerNorm(model_dim)
		self.drop1 = nn.Dropout(dropout)

		self.fc = FeedForward(model_dim, fc_dim, device)
		self.norm2 = nn.LayerNorm(model_dim)
		self.drop2 = nn.Dropout(dropout)

		self.to(device)

	def forward(self, X, mask=None):
		"""
		:param X: torch.Tensor, of shape [batch_size, T, d]
		:param mask: torch.Tensor, of shape [batch_size, T, T]
		:return:
		"""
		X = self.norm1(
			X + self.drop1(
				self.attn(X, X, X, mask)
			)
		)

		X = self.norm2(
			X + self.drop2(
				self.fc(X)
			)
		)
		return X

class Encoder(nn.Module):
	def __init__(self, model_dim, num_layers, num_heads=4, fc_dim=512, dropout=0.1, device='cpu'):
		super(Encoder, self).__init__()
		self.layers = nn.ModuleList([
			EncoderLayer(model_dim, num_heads, fc_dim, dropout, device)
			for _ in range (num_layers)
		])

	def forward(self, X, mask=None):
		"""
		:param X: torch.Tensor, of shape [batch_size, T, d]
		:param mask: torch.Tensor, of shape [batch_size, T]
		:return:
		"""
		for layer in self.layers:
			X = layer(X, mask[:,None,:])
		return X



def encoder_test():
	batch_size = 1
	T = 7
	dim = 5


	X = torch.randint(0, 5, [batch_size, T])
	embeding = torch.randint(0, 5, [batch_size, T, dim]).float()
	# pad_idx = torch.where(X==0)


	model = Encoder(dim, num_heads=3, num_layers=2, fc_dim=64)
	# model = EncoderLayer(dim, num_heads=1, fc_dim=64)
	y = model(embeding, (X!=0).int())
	print(y.shape)


if __name__ == '__main__':
	encoder_test()


