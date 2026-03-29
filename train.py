import torch
import random
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from component import board, architecture, agent, trainer, simulator, loss

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
	print(f"{'':10} {'as X':>6} {'as O':>6} {'Percentage':>10}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}" + f" {(wins[0] + wins[1]) / rounds * 100:>9.2f}%")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}" + f" {(draws[0] + draws[1]) / rounds * 100:>9.2f}%")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}" + f" {(losses[0] + losses[1]) / rounds * 100:>9.2f}%")
 
	return (wins, draws, losses)
	

if __name__ == "__main__":
	model = architecture.MLP64()
	eval_rounds = 1000

	# Play games and train
	epsilon = 0.7
	decay = 0.967
	games = []
	epochs = 10
	max_epochs = 100
	buffer = 32768
	optimizer = optim.Adam(model.parameters(), lr=0.001)
 
	sim = simulator.EpsilonGreedy(epsilon=epsilon, decay=decay)
	trn = trainer.DeepQ(optimizer, loss.MSEDual())
 
	print("Training the model with self-play")
	print(f"Self-play method: {sim.__class__.__name__}")
	print(f"Training method: {trn.__class__.__name__}")
	print(f"Loss function: {trn.loss_fn.__class__.__name__}")
	print("=" * 97)

	for i, rounds in enumerate([128, 256, 512, 1024, 2048, 4096, 8192, 8192, 8192, 8192, 8192, 8192, 8192]):
		print(f"Playing {rounds} games...")
		games.extend(sim.simulate(agent.Neural(model), rounds=rounds))
		games = games[-buffer:]
		trn.train(model, games, epochs=epochs, batch_size=128)
		epochs = min(epochs * 2, max_epochs)
		print("Evaluate against Minimax depth of 3:")
		evaluate(agent.Neural(model), agent.Minimax(depth=3, pruning=True), rounds=eval_rounds)
		print("=" * 97)
	print()

	# Evaluate after training
	print("Evaluate against Minimax depth of 4 after training:")
	evaluate(agent.Neural(model), agent.Minimax(depth=4, pruning=True), rounds=eval_rounds)

	# Save the trained model
	torch.save(model.state_dict(), "model/mlp64_deepq.pth")