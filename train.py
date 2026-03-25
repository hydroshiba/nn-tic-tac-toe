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

def train(model, games, epochs=1000, batch_size=256, optimizer=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device).train()
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
		total_loss = 0.0
		batches = 0
  
		for i in range(0, len(states), batch_size):
			idx = perm[i:i + batch_size]
			states_  = states[idx]   # [B, 9]
			actions_ = actions[idx]  # [B]
			rewards_ = rewards[idx]  # [B]
			
			predictions, value = model(states_)
			targets = predictions.detach().clone()
			targets[torch.arange(len(targets), device=device), actions_] = rewards_

			policy_loss = loss_fn(predictions, targets) * 9 # Scale loss to match reward range
			value_loss = loss_fn(value.squeeze(), rewards_)
			loss = policy_loss * 0.5 + value_loss * 0.5
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
			batches += 1
		
		if epoch % 10 == 0:
			tqdm.write(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss / batches:.4f}")

	# Print final loss
	final_loss = total_loss / batches if batches > 0 else 0.0
	print(f"Final training loss: {final_loss:.4f}")
	
	print("Evaluating against Minimax depth of 3:")
	model.to("cpu").eval()
	wins, draws, losses = evaluate(agent.Neural(model), agent.Minimax(depth=3), rounds=100)
	print(f"{'':10} {'as X':>6} {'as O':>6}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}")
	

if __name__ == "__main__":
	model = architecture.MLP64()
	agent1 = agent.Neural(model)
	agent2 = agent.Minimax(depth=4)
	eval_rounds = 500
	
	# Evaluate before training
	wins, draws, losses = evaluate(agent1, agent2, rounds=eval_rounds)
	print("Evaluate against Minimax depth of 4 before training:")
	print(f"{'':10} {'as X':>6} {'as O':>6}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6}")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}")
	print("=" * 97)

	# Play games and train
	epsilon = 0.7
	decay = 0.967
	games = []
	epochs = 10
	max_epochs = 100
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	for rounds in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
		print(f"Playing {rounds} games...")
		games.extend(play(model, rounds=rounds, epsilon=epsilon))
		epsilon *= decay
		train(model, games, epochs=epochs, batch_size=128, optimizer=optimizer)
		epochs = min(epochs * 2, max_epochs)
		print("=" * 97)

	# Evaluate after training
	agent1 = agent.Neural(model)

	wins, draws, losses = evaluate(agent1, agent2, rounds=eval_rounds)
	print("Evaluate against Minimax depth of 4 after training:")
	print(f"{'':10} {'as X':>6} {'as O':>6} {'Percentage':>10}")
	print(f"{'Wins':10} {wins[0]:>6} {wins[1]:>6} {(wins[0] + wins[1]) / eval_rounds * 100:>9.2f}%")
	print(f"{'Draws':10} {draws[0]:>6} {draws[1]:>6}" f"{(draws[0] + draws[1]) / eval_rounds * 100:>9.2f}%")
	print(f"{'Losses':10} {losses[0]:>6} {losses[1]:>6}" f"{(losses[0] + losses[1]) / eval_rounds * 100:>9.2f}%")

	# Save the trained model
	torch.save(model.state_dict(), "model/mlp64.pth")