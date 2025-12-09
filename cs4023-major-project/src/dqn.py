### This is for prototyping purposes, not full DDQN
### I want to test if the env works before starting with the complicated model
#!/usr/bin/env python3


import torch
import gymnasium as gym
import rospy
import math
import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple
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
            nn.Linear(46, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5)
        )

    def forward(self, x):
        # print(x.shape)
        # print(x.type())
        x = x.float()
        output = self.linear_relu_stack(x)
        return output


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# TODO: Add evalutaion metrics
# Total award per episode
# Success rate


class DQNAgent:
    def __init__(self, env):
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
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10

        # More Params
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999

        self.episode_durations = []

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
        print(action.shape)
        return action

    def update_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # Do we need this???

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device="cpu",
            dtype=torch.uint8,
        )

        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device="cpu")
        next_state_values[non_final_mask] = (
            self.target_network(non_final_next_states).max(1)[0].detach()
        )
        # Compute the expected Q values = Bellman Equation
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            cumulative_reward = 0
            for t in count():
                # Select and perform an action
                action = self.get_action(state)
                formatted_state = []
                formatted_state.extend(state[0])
                formatted_state.append(state[1])
                formatted_state.append(state[2])
                formatted_state.append(state[3])
                formatted_state = torch.tensor(formatted_state)
                # print(state)
                # print(action)
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += reward
                reward = torch.tensor([reward])

                formatted_nstate = []
                formatted_nstate.extend(next_state[0])
                formatted_nstate.append(next_state[1])
                formatted_nstate.append(next_state[2])
                formatted_nstate.append(next_state[3])
                formatted_nstate = torch.tensor(formatted_nstate)

                self.memory.push(formatted_state, action, formatted_nstate, reward)
                self.update_model()

                # Observe new state
                if not done:
                    state = next_state
                else:
                    self.episode_durations.append(t + 1)
                    break

                # Update the target network
                if episode % self.TARGET_UPDATE == 0:
                    self.target_network.load_state_dict(
                        self.policy_network.state_dict()
                    )
            result_str = "Episode " + str(episode) + " reward: " + str(cumulative_reward)
            rospy.loginfo(result_str)

        torch.save(self.target_network.state_dict())
