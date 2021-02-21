"""
Train an atari pong task.
"""
import argparse
import logging
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from queue import PriorityQueue, Queue
from typing import Dict, Final, List

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.ticker import MaxNLocator

from examples.utils import dump_json, set_logger, set_seed

logger = logging.getLogger(__name__)


def setup_output_dir(output_dir_path, args):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='data/model/cart_pole/')
    parser.add_argument('--time_step_plot', default='time_step.png')

    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--episode_count", type=int, default=2000)

    parser.add_argument("--algorithm", default='montecarlo')
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--first_visit", action='store_true')

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda_value", type=float, default=0.2)

    parser.add_argument("--render", action='store_true')

    parser.add_argument("--seed", type=int)

    args = parser.parse_args()

    return args


@dataclass
class ImageStateConverter:
    game_image_top: Final[int] = 34
    game_image_bottom: Final[int] = 194

    game_image_height: int = 160
    game_image_width: int = 160

    green_my_board: np.ndarray = np.array([92, 186, 92])
    white_ball: np.ndarray = np.array([236, 236, 236])
    brown_enemy_board: np.ndarray = np.array([213, 130, 74])
    env_action_to_network_action: Dict[int, int] = field(default_factory=lambda: {
        0: 0,
        2: 1,
        5: 2,
    })
    network_action_to_env_action: Dict[int, int] = field(default_factory=lambda: {
        0: 0,
        1: 2,
        2: 5,
    })
    prev_features: List[int] = field(default_factory=lambda: [0.] * 6 * 3)

    def convert_image_to_coords(self, image):
        self.game_image_height = 2000
        self.game_image_top = 2000

        is_my_board = np.all(
            image[self.game_image_top:self.game_image_bottom] == self.green_my_board, axis=-1)
        my_board_x = np.max(np.argmax(is_my_board, axis=1))
        my_board_y = np.max(np.argmax(is_my_board, axis=0))

        is_ball = np.all(
            image[self.game_image_top:self.game_image_bottom] == self.white_ball, axis=-1)
        ball_x = np.max(np.argmax(is_ball, axis=1))
        ball_y = np.max(np.argmax(is_ball, axis=0))

        is_enemy_board = np.all(
            image[self.game_image_top:self.game_image_bottom] == self.brown_enemy_board, axis=-1)
        enemy_board_x = np.max(np.argmax(is_enemy_board, axis=1))
        enemy_board_y = np.max(np.argmax(is_enemy_board, axis=0))

        return [
            my_board_x / self.game_image_height, my_board_y / self.game_image_height,
            ball_x / self.game_image_height, ball_y / self.game_image_height,
            enemy_board_x / self.game_image_height, enemy_board_y / self.game_image_height,
        ]

    def convert_observation(self, image):
        features = self.convert_image_to_coords(image)
        tensor_features = torch.tensor(
            self.prev_features + features, dtype=torch.float32)
        self.prev_features = self.prev_features[6:] + features
        return tensor_features

    def convert_reward(self, reward, prev_features, features):
        if reward == 0 and features[2] > 0.5 and features[2] - prev_features[2] == 0:
            return 1
        if reward > 0:
            return 100
        return reward

    def get_env_action_to_network_action(self, action):
        return self.env_action_to_network_action[action]

    def get_network_action_to_env_action(self, action):
        return self.network_action_to_env_action[action]

# class GameWindowNetwork(nn.Module):
#     def __init__(
#             self,
#             gpu_id=-1,
#         ):
#         super(GameWindowNetwork, self).__init__()
#         self.cnn1 = nn.Conv2d(3, 6, 3)
#         self.cnn2 = nn.Conv2d(3, 6, 3)

#         if gpu_id >= 0:
#             self.device = torch.device('cuda:{}'.format(gpu_id))
#         else:
#             self.device = torch.device('cpu')
#         self.to(self.device)

#     def forward(self, inputs):
#         _, state = self.encode(encoder_inputs)
#         decoder_outputs, _ = self.decode(decoder_inputs, state)

#         return decoder_outputs


class FeatureNeuralNetwork(nn.Module):
    def __init__(
        self,
        gpu_id=-1,
    ):
        super(FeatureNeuralNetwork, self).__init__()
        self.linear = nn.Linear(27, 1)

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, inputs):
        features, action = inputs
        features = torch.cat([features, F.one_hot(
            torch.tensor(action), num_classes=3)])
        output = self.linear(features)

        return output


def get_network_action(network, features, epsilon):
    network_action = np.argmax(
        [network((features, a)).data.numpy() for a in range(3)])
    if np.random.uniform() <= epsilon:
        return np.random.randint(0, 3)
    return network_action


def get_max_q_value(network, features):
    return np.max([network((features, a)).data.numpy() for a in range(3)])


def train_network(q_network, target_network, optimizer, prev_features, action, reward, features):
    outputs = q_network((prev_features, action))
    targets = get_max_q_value(target_network, features)

    loss = nn.MSELoss()(outputs, torch.tensor(
        [targets + reward], dtype=torch.float32))
    loss.backward()
    optimizer.step()

    return loss.data.numpy()


def q_learning(
    env: gym.wrappers.time_limit.TimeLimit,
    n_step: int = 1,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    episode_count: int = 2000,
    render: bool = False,
):
    image_state_converter = ImageStateConverter()

    q_network = FeatureNeuralNetwork()
    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)

    model = dict()
    memory = set()

    q_network.train()
    for i in range(episode_count):
        curr_epsilon = (episode_count - i) / episode_count * epsilon

        prev_observation = env.reset()
        prev_features = image_state_converter.convert_observation(
            prev_observation)
        target_network = deepcopy(q_network)
        for t in range(10000):
            if render and i >= 100:
                # if render:
                env.render()

            network_action = get_network_action(
                q_network, prev_features, curr_epsilon)
            env_action = image_state_converter.get_network_action_to_env_action(
                network_action)

            observation, reward, done, info = env.step(env_action)
            features = image_state_converter.convert_observation(observation)
            reward = image_state_converter.convert_reward(
                reward, prev_features, features)

            loss = train_network(
                q_network, target_network, optimizer,
                prev_features, network_action, reward, features,
            )

            model[(prev_features, network_action)] = (features, reward)
            memory.add((prev_features, network_action))

            for memory_prev_features, memory_action in random.sample(memory, min(n_step, len(memory))):
                memory_features, memory_reward = model[(
                    memory_prev_features, memory_action)]
                train_network(
                    q_network, target_network, optimizer,
                    memory_prev_features, memory_action, memory_reward, memory_features,
                )

            prev_features = features

            if done:
                print(f'Episode {i + 1} finished after {t + 1} timesteps')
                break


def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    env = gym.make('PongDeterministic-v4')
    q_learning(env, args.n_step, args.gamma, args.epsilon,
               args.episode_count, args.render)
    env.close()


if __name__ == '__main__':
    run()
