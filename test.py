import torch
import random
from tqdm import tqdm
from component import board, architecture, agent

def evaluate(agent1, agent2, rounds=1000):
	wins = [0, 0]
	draws = [0, 0]

	for _ in tqdm(range(0, rounds, 2)):
		b = board.Board()
		player = 1
		while not b.terminal():
			if player == 1: move = agent1.play(b, player)
			else: move = agent2.play(b, player)
			b.make_move(move, player)
			player = -player

		result = b.evaluate()
		if result == 1: wins[0] += 1
		elif result == 0: draws[0] += 1

		b = board.Board()
		player = 1
		while not b.terminal():
			if player == 1: move = agent2.play(b, player)
			else: move = agent1.play(b, player)
			b.make_move(move, player)
			player = -player

		result = b.evaluate()
		if result == -1: wins[1] += 1
		elif result == 0: draws[1] += 1

	losses = [rounds // 2 - wins[0] - draws[0], rounds // 2 - wins[1] - draws[1]]
	return (wins, draws, losses)

if __name__ == "__main__":
	model = architecture.MLP64()
	model.load_state_dict(torch.load("model/mlp64_deepq.pth"))
	model.eval()

	minimax = agent.Minimax(depth=0, evaluation=model)

	for depth in range(1, 7):
		print(f"Evaluating pure neural network (MLP 9 -> 64 -> 64 -> 1) against Minimax{depth} with pruning...")
		wins, draws, losses = evaluate(minimax, agent.Minimax(depth=depth, pruning=True), rounds=1000)
		print(f"{'':10} {'as X':>6} {'as O':>6}")
		print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}")
		print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}")
		print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}")