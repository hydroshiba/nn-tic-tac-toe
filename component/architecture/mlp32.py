import torch
from torch import nn, optim

class MLP32(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden = nn.Linear(9, 32)
		self.act = nn.ReLU()

		self.policy = nn.Linear(32, 9)
		self.value = nn.Linear(32, 1)

	def forward(self, x):
		x = self.act(self.hidden(x))
		return self.policy(x), nn.Tanh()(self.value(x))

	def evaluate(self, x):
		with torch.no_grad():
			_, value = self.forward(x)
		return value.item()

	def q_value(self, x):
		with torch.no_grad():
			policy, _ = self.forward(x)
		return policy