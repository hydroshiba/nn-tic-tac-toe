import torch
import torch.nn.functional as F
from torch import nn

class MSEDual:
	def __init__(self):
		self.mse = nn.MSELoss()

	def __call__(self, predictions, moves, targets, values, results):
		policies = predictions[range(len(moves)), moves]
		policy_loss = self.mse(policies, targets)
		value_loss = self.mse(values.squeeze(-1), results)
		return policy_loss * 0.5 + value_loss * 0.5

# Do not use, currently broken
class PolicyGradient:
	def __init__(self):
		self.mse = nn.MSELoss()
		self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

	def __call__(self, predictions, moves, targets, values, results):
		policy_loss = (self.cross_entropy(predictions, moves) * results).mean()
		value_loss = self.mse(values.squeeze(-1), results)
		return policy_loss * 0.5 + value_loss * 0.5