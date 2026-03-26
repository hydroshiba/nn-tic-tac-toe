import torch
from torch import nn, optim

class MLP64(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden1 = nn.Linear(9, 64)
		self.hidden2 = nn.Linear(64, 64)
		self.act = nn.ReLU()

		self.policy_head = nn.Linear(64, 9)
		self.value_head = nn.Linear(64, 1)

	def forward(self, x):
		x = self.act(self.hidden1(x))
		x = self.act(self.hidden2(x))
		return self.policy_head(x), nn.Tanh()(self.value_head(x))
	
	def evaluate(self, x):
		with torch.no_grad():
			_, val = self.forward(x)
		return val.item()

	def policy(self, x):
		with torch.no_grad():
			pol, _ = self.forward(x)
		return pol