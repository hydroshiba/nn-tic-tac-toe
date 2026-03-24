import torch
import random
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from component import board, architecture, agent

def play(model, rounds=1000, epsilon=0.1):
	games = []

	for _ in tqdm(range(rounds)):
		logs = []
		b = board.Board()
		player = 1

		while not b.terminal():
			canonical = b.board * player
			policy = model.q_value(canonical)
			move = [i for i in range(9) if b.board[i] == 0][torch.argmax(policy[b.board == 0])]
			if random.random() < epsilon:
				move = random.choice([i for i in range(9) if b.board[i] == 0])

			logs.append((canonical.clone(), move, player))
			b.make_move(move, player)
			player = -player

		result = b.evaluate()
		games.append([(state, action, result * player) for state, action, player in logs])
	
	return games

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

def train(model, games, epochs=1000):
	device = torch.device("cuda" if torch.cuda.is_available() and len(games) > 5000 else "cpu")
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	loss_fn = nn.MSELoss()

	# Flatten all steps across all games into batched tensors once
	all_states, all_actions, all_rewards = [], [], []
	for game in games:
		for state, action, reward in game:
			all_states.append(state)
			all_actions.append(action)
			all_rewards.append(reward)

	states  = torch.stack(all_states).to(device)                            # [N, 9]
	actions = torch.tensor(all_actions, dtype=torch.long, device=device)    # [N]
	rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device) # [N]

	for epoch in tqdm(range(epochs)):
		perm     = torch.randperm(len(states), device=device)
		states_  = states[perm]
		actions_ = actions[perm]
		rewards_ = rewards[perm]

		predictions, _ = model(states_)
		targets = predictions.detach().clone()
		targets[torch.arange(len(targets), device=device), actions_] = rewards_

		loss = loss_fn(predictions, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		tqdm.write(f"Epoch {epoch + 1}/{epochs} — loss: {loss.item():.6f}")

	print("Evaluating against self play:")
	wins, draws, losses = evaluate(agent.Neural(model), agent.Neural(model), rounds=100)
	print(f"{'':10} {'as X':>6} {'as O':>6}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}")
	

if __name__ == "__main__":
	model = architecture.MLP32()
	agent1 = agent.Neural(model)
	agent2 = agent.Minimax(depth=3)
	
	# Evaluate before training
	wins, draws, losses = evaluate(agent1, agent2, rounds=1000)
	print("Evaluate against Minimax depth of 3 before training:")
	print(f"{'':10} {'as X':>6} {'as O':>6}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}")
	print("=" * 97)

	# Play games and train
	
	for rounds in [100, 1000, 5000, 10000, 20000]:
		print(f"Playing {rounds} games...")
		games = play(model, rounds=rounds)
		train(model, games, epochs=100)
		print("=" * 97)

	# Evaluate after training
	agent1 = agent.Neural(model)

	wins, draws, losses = evaluate(agent1, agent2, rounds=1000)
	print("Evaluate against Minimax depth of 3 after training:")
	print(f"{'':10} {'as X':>6} {'as O':>6}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}")

	# Save the trained model
	torch.save(model.state_dict(), "model/mlp32.pth")