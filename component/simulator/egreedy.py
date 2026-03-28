import torch
import torch.nn.functional as F

from tqdm import tqdm
from component import board

class EpsilonGreedy:
	def __init__(self, epsilon=0.1, decay=1):
		self.epsilon = epsilon
		self.decay = decay

	def simulate(self, agent, rounds=1000):
		games = []

		for _ in tqdm(range(rounds)):
			logs = []
			b = board.Board()
			player = 1

			while not b.terminal():
				legal_moves = b.legal_moves()
				if torch.rand(1).item() < self.epsilon: move = legal_moves[torch.randint(len(legal_moves), (1,)).item()]
				else: move = agent.play(b, player)
				policy = F.one_hot(torch.tensor(move), num_classes=len(b.board)).float()

				logs.append((b.board.clone(), player, move, policy))
				b.make_move(move, player)
				player = -player

			result = b.evaluate()
			games.append([(log[0], log[1], log[2], log[3], result) for log in logs])
		
		self.epsilon *= self.decay
		return games