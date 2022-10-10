import numpy as np
import torch
from torch import nn

class AttentionHead(nn.Module):
	def __init__(self, q_dim, k_dim, v_dim, embed_dim, device='cpu'):
		super(AttentionHead, self).__init__()
		self.embed_dim = embed_dim
		self.minf = -1e9
		self.q_linear = nn.Linear(q_dim, embed_dim)
		self.k_linear = nn.Linear(k_dim, embed_dim)
		self.v_linear = nn.Linear(v_dim, embed_dim)

		self.device = device
		self.to(device)

	def forward(self, q, k, v, mask=None):
		"""
		:param q: [batch_size, T1, q_dim]
		:param k: [batch_size, T2, k_dim]
		:param v: [batch_size, T2, v_dim]
		:param mask: [batch_size, T2, T2] or [batch_size, 1, T2]
		:return:
		"""
		Q = self.q_linear(q)  # [batch_size, T1, embed_dim]
		K = self.k_linear(k)  # [batch_size, T2, embed_dim]
		V = self.v_linear(v)  # [batch_size, T2, embed_dim]
		E = torch.bmm(Q, torch.transpose(K, 1, 2)) / np.sqrt(self.embed_dim)  # [batch_size, T1, T2]
		if mask is not None:
			mask = mask.to(self.device)
			# print(mask.shape, E.shape)
			E = mask * E + (1. - mask) * self.minf
			# print(mask)
		A = torch.softmax(E, dim=-1)
		# print(A)
		return torch.bmm(A, V)

class MultiHeadAttention(nn.Module):
	def __init__(self, q_dim, k_dim, v_dim, embed_dim, num_heads, device='cpu'):
		super(MultiHeadAttention, self).__init__()
		head_dim = embed_dim // num_heads

		self.heads = nn.ModuleList([
			AttentionHead(q_dim, k_dim, v_dim, head_dim, device) for _ in range(num_heads)
		])
		self.linear = nn.Linear(head_dim * num_heads, embed_dim)
		self.to(device)

	def forward(self, q, k, v, mask=None):
		"""
		:param q: [batch_size, T1, q_dim]
		:param k: [batch_size, T2, k_dim]
		:param v: [batch_size, T2, v_dim]
		:param mask: [batch_size, T2]
		:return:
		"""
		ys = torch.cat([
			head(q, k, v, mask) for head in self.heads
		], dim=-1)
		ys = self.linear(ys)
		return ys

class PositionalEncoding(nn.Module):
	def __init__(self, dim, max_length=None, freq_inv=1e5, device='cpu'):
		super(PositionalEncoding, self).__init__()
		assert dim % 2 == 0
		self.dim = dim
		self.max_length = max_length
		self.freq_inv = freq_inv
		self.device = device

		if max_length is not None:
			self.coss, self.sins = self.get_matrix(max_length)

	def get_matrix(self, T):
		assert self.dim % 2 == 0
		ts = torch.repeat_interleave(
			torch.arange(0, T)[None, :], self.dim//2, dim=0
		).to(self.device)

		ks = torch.repeat_interleave(
			torch.arange(0, self.dim//2)[:, None], T, dim=1
		).to(self.device)

		phis = ts / torch.pow(self.freq_inv, 2 * ks / self.dim)

		coss = torch.cos(phis).T # [seq_length, dim//2]
		sins = torch.sin(phis).T # [seq_length, dim//2]

		# for r in coss.detach().numpy().T[:10]:
		# 	plt.plot(r, 'r--')
		# for r in sins.detach().numpy().T[:10]:
		# 	plt.plot(r, 'b--')
		# plt.show()

		return coss, sins


	def forward(self, X):
		"""
		:param X: torch.Tensor, of shape [batch_size, seq_length, dim]
		:return:
		"""
		batch_size, seq_length, dim = X.shape
		if self.max_length is None:
			coss, sins = self.get_matrix(seq_length)
		else:
			coss, sins = self.coss[:seq_length, :], self.sins[:seq_length,:]

		X[:, :, 0::2] += coss
		X[:, :, 1::2] += sins
		return X

class FeedForward(nn.Module):
	def __init__(self, dim_in, fc_dim, device='cpu'):
		super(FeedForward, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(dim_in, fc_dim),
			nn.ReLU(),
			nn.Linear(fc_dim, dim_in)
		)

		self.to(device)

	def forward(self, X):
		return self.net(X)

def self_attn_test():
	batch_size = 1
	T1 = 3
	T2 = 6
	dim = 5
	source = torch.randn(batch_size, T1, dim)
	target = torch.randn(batch_size, T2, dim)

	mask = torch.randint(0,3, size=[batch_size, T2])
	mask = (mask == 2).int()

	model = MultiHeadAttention(dim, dim, dim, dim, 1)
	model(source, target, target, mask)

# self_attn_test()

