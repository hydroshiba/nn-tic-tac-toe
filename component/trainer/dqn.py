import torch
import copy
from tqdm import tqdm

class DQN:
	def __init__(self, optimizer, loss_fn, gamma=0.95, target_update_freq=5):
		self.optimizer = optimizer
		self.loss_fn = loss_fn
		self.gamma = gamma
		self.target_update_freq = target_update_freq

	def train(self, model, games, epochs=1000, batch_size=256):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device).train()

		target_model = copy.deepcopy(model)
		target_model.eval()

		states, players, moves, next_states, dones, results = [], [], [], [], [], []
		for game in games:
			for i, (state, player, move, policy, result) in enumerate(game):
				terminal = (i == len(game) - 1)
				next_state = torch.zeros(len(state)) if terminal else game[i + 1][0]
				states.append(state)
				players.append(player)
				moves.append(move)
				next_states.append(next_state)
				dones.append(1.0 if terminal else 0.0)
				results.append(result * player)

		states = torch.stack(states).to(device)
		players = torch.tensor(players).to(device).float()
		moves = torch.tensor(moves).to(device).long()
		next_states = torch.stack(next_states).to(device)
		dones = torch.tensor(dones).to(device).float()
		results = torch.tensor(results).to(device).float()

		for epoch in tqdm(range(epochs)):
			if epoch % self.target_update_freq == 0:
				target_model.load_state_dict(model.state_dict())

			perm = torch.randperm(len(states))
			total_loss = 0.0
			batches = 0

			for i in range(0, len(states), batch_size):
				idx = perm[i:i + batch_size]
				batch_states = states[idx] * players[idx].unsqueeze(1)
				batch_moves = moves[idx]
				batch_next_states = next_states[idx] * (-players[idx]).unsqueeze(1)
				batch_dones = dones[idx]
				batch_results = results[idx]

				# Bellman target: terminal -> result; non-terminal -> gamma * -max Q(s')
				# Negate opponent's best Q because the game is zero-sum
				with torch.no_grad():
					next_preds, _ = target_model(batch_next_states)
					illegal_mask = (batch_next_states != 0)
					next_preds[illegal_mask] = float('-inf')
					max_next_q = next_preds.max(dim=1).values
					q_values = batch_results * batch_dones - self.gamma * max_next_q * (1 - batch_dones)

				predictions, values = model(batch_states)
				loss = self.loss_fn(predictions, batch_moves, q_values, values, batch_results)

				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				self.optimizer.step()

				total_loss += loss.item()
				batches += 1

			if epoch % 10 == 0:
				tqdm.write(f"Epoch {epoch + 1}/{epochs} — loss: {total_loss / batches:.4f}")

		final_loss = total_loss / batches if batches > 0 else 0.0
		print(f"Final training loss: {final_loss:.4f}")
