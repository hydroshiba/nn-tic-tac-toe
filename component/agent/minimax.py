class Minimax:
	def __init__(self, depth):
		self.depth = depth
	
	def minimax(self, board, depth, player):
		if depth == 0 or board.terminal():
			return board.evaluate() * player
		
		eval = float('-inf')
		for move in board.legal_moves():
			board.make_move(move, player)
			eval = max(eval, -self.minimax(board, depth - 1, -player))
			board.undo_move(move)

		return eval

	def play(self, board, player):
		eval = float('-inf')
		best_move = None
		
		for move in board.legal_moves():
			board.make_move(move, player)
			move_eval = -self.minimax(board, self.depth - 1, -player)
			board.undo_move(move)
   
			if move_eval > eval:
				eval = move_eval
				best_move = move
		
		return best_move