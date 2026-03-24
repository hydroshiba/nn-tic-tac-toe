import torch
from torch import nn, optim

class MLP64(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden1 = nn.Linear(9, 64)
		self.hidden2 = nn.Linear(64, 64)
		self.act = nn.ReLU()

		self.policy = nn.Linear(64, 9)
		self.value = nn.Linear(64, 1)

	def forward(self, x):
		x = self.act(self.hidden1(x))
		x = self.act(self.hidden1(x))
		return self.policy(x), nn.Tanh()(self.value(x))
	
	def evaluate(self, x):
		with torch.no_grad():
			_, value = self.forward(x)
		return value.item()

	def q_value(self, x):
		with torch.no_grad():
			policy, _ = self.forward(x)
		return policy