import torch
import numpy as np

class Board:
	def __init__(self):
		self.board = torch.zeros(9)

	def make_move(self, move, player):
		if self.board[move] == 0:
			self.board[move] = player
			return True
		return False

	def undo_move(self, move):
		self.board[move] = 0

	def legal_moves(self):
		moves = [i for i in range(9) if self.board[i] == 0]
		np.random.shuffle(moves)
		return moves

	def terminal(self):
		return self.evaluate() != 0 or len(self.legal_moves()) == 0

	def evaluate(self):
		board_2d = self.board.view(3, 3)
		for i in range(3):
			if torch.all(board_2d[i] == 1) or torch.all(board_2d[:, i] == 1):
				return 1
			if torch.all(board_2d[i] == -1) or torch.all(board_2d[:, i] == -1):
				return -1
		if torch.all(torch.diag(board_2d) == 1) or torch.all(torch.diag(torch.fliplr(board_2d)) == 1):
			return 1
		if torch.all(torch.diag(board_2d) == -1) or torch.all(torch.diag(torch.fliplr(board_2d)) == -1):
			return -1
		return 0

	def view(self):
		return self.board.view(3, 3)