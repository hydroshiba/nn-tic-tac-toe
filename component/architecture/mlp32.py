import torch
from torch import nn, optim

class MLP32(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(9, 32)
		self.act = nn.ReLU()

		self.policy_head = nn.Linear(32, 9)
		self.value_head = nn.Linear(32, 1)

	def forward(self, x):
		x = self.act(self.hidden(x))
		return self.policy_head(x), nn.Tanh()(self.value_head(x))

	def evaluate(self, x):
		with torch.no_grad():
			_, val = self.forward(x)
		return val.item()

	def policy(self, x):
		with torch.no_grad():
			pol, _ = self.forward(x)
		return pol