import torch

class Neural:
	def __init__(self, model):
		self.model = model
		self.model.eval()
	
	def play(self, board, player):
		move_values = self.model.q_value(board.board * player)
		legal_moves = board.legal_moves()
		best_move = [move for move in legal_moves if move_values[move] == move_values[legal_moves].max()][0]
		return best_move