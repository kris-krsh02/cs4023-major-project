### This is for prototyping purposes, not full DDQN
### I want to test if the env works before starting with the complicated model
#!/usr/bin/env python3


import torch
import gymnasium as gym
import rospy
import math
import random
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
# from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(27, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5)
        )

    def forward(self, x):
        x = x.float()
        output = self.linear_relu_stack(x)
        return output


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(*args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# TODO: Add evalutaion metrics
# Total award per episode
# Success rate


class DQNAgent:
    def __init__(self, env, gamma, eps_decay):
        self.env = env
        self.policy_network = NeuralNetwork()
        self.target_network = NeuralNetwork()
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.RMSprop(self.policy_network.parameters())
        self.memory = ReplayMemory(1000)
        self.steps_done = 0

        # Epsilon values
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = 20

        # More Params
        self.BATCH_SIZE = 64
        self.GAMMA = gamma

        self.episode_steps = []
        self.cumulative_rewards = []
        self.all_loss = []
        self.outcomes = []

        self.model_name = f"g_{gamma}_eps_{eps_decay}"

        # Window size for moving averages
        self.window_size = 20

    def get_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1.0 * self.steps_done / self.EPS_DECAY
        )
        action=None

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                formatted_state = []
                formatted_state.extend(state[0])
                formatted_state.append(state[1])
                formatted_state.append(state[2])
                formatted_state.append(state[3])
                formatted_state = torch.tensor(formatted_state)
                action = self.policy_network(formatted_state).max(0)[1].reshape(1, 1)
        else:
            action = torch.tensor([[random.randrange(5)]], device="cpu", dtype=torch.long)
        return action

    def update_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # Do we need this???

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device="cpu",
            dtype=torch.bool,
        )

        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device="cpu")

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        
        # Compute the expected Q values = Bellman Equation
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        self.all_loss.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(1, num_episodes + 1):
            state, _ = self.env.reset()
            cumulative_reward = 0
            outcome = "FAIL"
            for t in count():
                # Select and perform an action
                action = self.get_action(state)
                formatted_state = []
                formatted_state.extend(state[0])
                formatted_state.append(state[1])
                formatted_state.append(state[2])
                formatted_state.append(state[3])
                formatted_state = torch.tensor(formatted_state)
                next_state, reward, done, success, info = self.env.step(action)
                cumulative_reward += reward
                reward = torch.tensor([reward])

                formatted_nstate = []
                formatted_nstate.extend(next_state[0])
                formatted_nstate.append(next_state[1])
                formatted_nstate.append(next_state[2])
                formatted_nstate.append(next_state[3])
                formatted_nstate = torch.tensor(formatted_nstate)

                self.memory.push([formatted_state, action, formatted_nstate, reward])
                self.update_model()

                # Observe new state
                if not done:
                    state = next_state
                else:
                    self.episode_steps.append(t + 1)
                    if success:
                        self.outcomes.append(1)
                        outcome = "SUCCESS"
                    else:
                        self.outcomes.append(0)
                        break
                    break

                # Update the target network
                if episode % self.TARGET_UPDATE == 0:
                    self.target_network.load_state_dict(
                        self.policy_network.state_dict()
                    )
            
            result_str = "EPISODE " + str(episode).zfill(4) + " " + outcome + ": " + str(cumulative_reward) + ", " + str(self.episode_steps[-1])

            rospy.loginfo(result_str)
            self.cumulative_rewards.append(cumulative_reward)

        # Save metrics and generate plots
        self.plot_rewards()
        self.plot_loss()
        self.plot_num_steps()
        self.plot_success()
        filename = f"models/model_{str(self.GAMMA)[0:]}_{self.EPS_DECAY}.pth"
        torch.save(self.target_network.state_dict(), filename)


    # Generate plots and csv for rewards metrics
    def plot_rewards(self):
        # episode_numbers: 1-indexed list of episode indices
        episode_numbers = list(range(1, len(self.cumulative_rewards) + 1))

        # Calculate moving averages, #1-indexed
        moving_avgs = []
        for i in episode_numbers:
            if i < self.window_size:
                # average the first i rewards
                moving_avgs.append(sum(self.cumulative_rewards[0:i])/(i)) 
            else:
                # average the last 'window_size' number of rewards
                moving_avgs.append(sum(self.cumulative_rewards[i-self.window_size:i])/self.window_size) 

        # Plot cumulative reward and moving average reward
        plt.plot(episode_numbers, self.cumulative_rewards, label='Cumulative Reward')
        plt.plot(episode_numbers, moving_avgs, label='Average Reward')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.title(f"Model Gamma: {self.GAMMA} Eps Decay: {self.EPS_DECAY}")
        plt.savefig(f"plots/reward_{str(self.GAMMA)}_{self.EPS_DECAY}.png")
        plt.close()

        # Save cumulative reward
        with open(f'data/c_reward_{str(self.GAMMA)}_{self.EPS_DECAY}.csv', 'w', newline='') as f:
            data = zip(episode_numbers, self.cumulative_rewards)
            writer = csv.writer(f)
            writer.writerows(data)

        # Save moving average reward
        with open(f'data/m_reward_{str(self.GAMMA)}_{self.EPS_DECAY}.csv', 'w', newline='') as f:
            data = zip(episode_numbers, moving_avgs)
            writer = csv.writer(f)
            writer.writerows(data)
 
    # Generate plots and csv for loss metric
    def plot_loss(self):
        num_steps = list(range(1, len(self.all_loss) + 1))
        plt.plot(num_steps, self.all_loss)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"Model Gamma: {self.GAMMA} Eps Decay: {self.EPS_DECAY}")
        plt.savefig(f"plots/loss_{str(self.GAMMA)}_{self.EPS_DECAY}.png")
        plt.close()

        with open(f'data/loss_{str(self.GAMMA)}_{self.EPS_DECAY}.csv', 'w', newline='') as f:
            data = zip(num_steps, self.all_loss)
            writer = csv.writer(f)
            writer.writerows(data)

    # Generate plots and save num steps metric
    def plot_num_steps(self):
        # episode_numbers: 1-indexed list of episode indices
        episode_numbers = list(range(1, len(self.episode_steps) + 1))

        # Calculate moving averages, #1-indexed
        moving_avgs = []
        for i in episode_numbers:
            if i < self.window_size:
                # average the first i num steps
                moving_avgs.append(sum(self.episode_steps[0:i])/(i)) 
            else:
                # average the last 'window_size' number of num steps
                moving_avgs.append(sum(self.episode_steps[i-self.window_size:i])/self.window_size) 

        # Plot num steps and moving average num steps
        plt.plot(episode_numbers, self.episode_steps, label='Episode Steps')
        plt.plot(episode_numbers, moving_avgs, label='Average Steps')
        plt.xlabel("Episode")
        plt.ylabel("Number of Steps")
        plt.legend()
        plt.title(f"Model Gamma: {self.GAMMA} Eps Decay: {self.EPS_DECAY}")
        plt.savefig(f"plots/num_steps_{str(self.GAMMA)}_{self.EPS_DECAY}.png")
        plt.close()

        # Save num steps
        with open(f'data/c_num_steps_{str(self.GAMMA)}_{self.EPS_DECAY}.csv', 'w', newline='') as f:
            data = zip(episode_numbers, self.episode_steps)
            writer = csv.writer(f)
            writer.writerows(data)

        # Save moving average num steps
        with open(f'data/m_num_steps_{str(self.GAMMA)}_{self.EPS_DECAY}.csv', 'w', newline='') as f:
            data = zip(episode_numbers, moving_avgs)
            writer = csv.writer(f)
            writer.writerows(data)

    # Generate plots and save success rate metric
    def plot_success(self):
        # episode_numbers: 1-indexed list of episode indices
        episode_numbers = list(range(1, len(self.outcomes) + 1))
        
        # Calculate success rates
        num_success = 0
        success_rates = []
        for i in episode_numbers:
            num_success += self.outcomes[i-1]
            success_rates.append(num_success/i)

        # Plot success rates
        plt.plot(episode_numbers, success_rates, label='Success Rate')
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title(f"Model Gamma: {self.GAMMA} Eps Decay: {self.EPS_DECAY}")
        plt.savefig(f"plots/success_rate_{str(self.GAMMA)}_{self.EPS_DECAY}.png")
        plt.close()

        # Save success rates
        with open(f'data/success_rates_{str(self.GAMMA)}_{self.EPS_DECAY}.csv', 'w', newline='') as f:
            data = zip(episode_numbers, success_rates)
            writer = csv.writer(f)
            writer.writerows(data)