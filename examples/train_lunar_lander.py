"""
Train a lunar lander task.
"""
# state:
#       cart's position: -2.4 - 2.4
#       cart's velocity: -3.0 - 3.0
#       bar's angle (radian):   -2.4 - 2.4
#       bar's angular velocity(radian): -2.0 - 2.0
# termination:
#       cart's position > 2.4 or cart's position< -2.4
#       bar's angle > 15 / 2 / pi or bar's angle < 15 / 2 / pi

import os
import argparse
import logging
from itertools import product
import random
from copy import deepcopy
from queue import Queue, PriorityQueue
import heapq
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from examples.utils import set_seed, set_logger, dump_json

logger = logging.getLogger(__name__)

def setup_output_dir(output_dir_path, args):
    os.makedirs(output_dir_path, exist_ok=True)
    dump_json(args, os.path.join(output_dir_path, 'args.json'))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='data/model/mountain_car/')
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

def plot(values, label, figure_path):
    episode_numbers = np.arange(0, len(values))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(episode_numbers, np.array(values)[episode_numbers], label='time_step')
    ax.set_xlabel('Episode')
    ax.set_ylabel(label)
    ax.legend()

    x_ax = ax.get_xaxis()
    x_ax.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(figure_path)
    plt.close()

class LunarLanderStateConverter:
    def __init__(self, episode_count, epsilon=0.2):
        self.episode_count = episode_count
        self.initial_epsilon = epsilon
        self.current_episode = 0

        self.min_pos_x = -1.0
        self.max_pos_x = 0.8
        self.pos_x_size = 6

        self.min_pos_y = -0.2
        self.max_pos_y = 1.5
        self.pos_y_size = 12

        self.min_v_x = -1.0
        self.max_v_x = 1.0
        self.v_x_size = 6

        self.min_v_y = -1.5
        self.max_v_y = 0.3
        self.v_y_size = 12

        self.min_angle = -1.2
        self.max_angle = 1.2
        self.angle_size = 6

        self.min_angular_velocity = -3.0
        self.max_angular_velocity = 3.0
        self.angular_velocity_size = 6

        self.action_num = 4

    @property
    def epsilon(self):
        limit_epoch = 30
        if self.current_episode < limit_epoch:
            return self.initial_epsilon * (1.0 - self.current_episode / limit_epoch)
        return self.initial_epsilon * 0.1

    def next(self):
        self.current_episode += 1

    def state_to_index(self, state):
        def bins(clip_min, clip_max, num):
            return np.linspace(clip_min, clip_max, num + 1)[1:-1]

        pos_x, pos_y, v_x, v_y, ang, ang_v, leg1, leg2 = state

        pos_x_idx = np.digitize(pos_x, bins=bins(
            self.min_pos_x, self.max_pos_x, self.pos_x_size))
        pos_y_idx = np.digitize(pos_y, bins=bins(
            self.min_pos_y, self.max_pos_y, self.pos_y_size))
        v_x_idx = np.digitize(v_x, bins=bins(
            self.min_v_x, self.max_v_x, self.v_x_size))
        v_y_idx = np.digitize(v_y, bins=bins(
            self.min_v_y, self.max_v_y, self.v_y_size))
        ang_idx = np.digitize(ang, bins=bins(
            self.min_angle, self.max_angle, self.angle_size))
        ang_v_idx = np.digitize(ang_v, bins=bins(
            self.min_angular_velocity, self.max_angular_velocity, self.angular_velocity_size))

        return pos_x_idx, pos_y_idx, v_x_idx, v_y_idx, ang_idx, ang_v_idx, int(leg1), int(leg2)

    def get_initial_q_value(self):
        q_value = np.random.uniform(low=0, high=1, size=(
            self.pos_x_size,
            self.pos_y_size,
            self.v_x_size,
            self.v_y_size,
            self.angle_size,
            self.angular_velocity_size,
            2,
            2,
            self.action_num,
        ))
        q_value[0] = 0
        q_value[-1] = 0
        q_value[:, :, 0] = 0
        q_value[:, :, -1] = 0

        return q_value

def get_action(q_value, state_converter, state):
    index = state_converter.state_to_index(state)
    action = np.argmax(q_value[index])
    if np.random.uniform() <= state_converter.epsilon:
        action = np.random.randint(0, state_converter.action_num)

    return action


# def get_action(q_value, state_converter, state):
#     position, velocity = state

#     return 1 if velocity >= 0 else 0


class FeatureNeuralNetwork(nn.Module):
    def __init__(
        self,
        gpu_id=-1,
    ):
        super(FeatureNeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(12, 16)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(16, 1)

        if gpu_id >= 0:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, inputs):
        state, action = inputs
        state = torch.tensor(state, dtype=torch.float32)
        features = torch.cat([
            state,
            F.one_hot(
                torch.tensor(action),
                num_classes=4,
            )
        ])

        h = self.linear1(features)
        h = self.relu(h)
        output = self.linear2(h)

        return output


def get_network_action(network, state, epsilon):
    network_action = np.argmax(
        [network((state, a)).data.numpy() for a in range(4)])
    if np.random.uniform() <= epsilon:
        return np.random.randint(0, 4)
    return network_action



def get_max_q_value(network, features):
    return np.max([network((features, a)).data.numpy() for a in range(4)])


def train_network(q_network, target_network, optimizer, prev_features, action, reward, features):
    outputs = q_network((prev_features, action))
    targets = get_max_q_value(target_network, features)

    loss = nn.MSELoss()(outputs, torch.tensor(
        [targets + reward], dtype=torch.float32))
    loss.backward()
    optimizer.step()

    return loss.data.numpy()


def dqn(
    env: gym.wrappers.time_limit.TimeLimit,
    n_step: int = 1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    episode_count: int = 2000,
    render: bool = False,
):
    q_network = FeatureNeuralNetwork()
    optimizer = optim.RMSprop(q_network.parameters(), lr=1e-3)

    model = dict()
    memory = set()

    q_network.train()
    for i in range(episode_count):
        curr_epsilon = (100 - i) / 100 * epsilon
        curr_epsilon = max(curr_epsilon, 0.1)

        prev_observation = env.reset()
        for t in range(10000):
            target_network = deepcopy(q_network)
            if render:
                env.render()

            env_action = get_network_action(
                q_network, prev_observation, curr_epsilon)

            observation, reward, done, info = env.step(env_action)
            if abs(reward) > 5.0:
                print(reward)

            loss = train_network(
                q_network, q_network, optimizer,
                prev_observation, env_action, reward, observation,
            )

            model[(tuple(prev_observation.tolist()), env_action)] = (observation, reward)
            memory.add((tuple(prev_observation.tolist()), env_action))

            for memory_prev_features, memory_action in random.sample(memory, min(n_step, len(memory))):
                memory_features, memory_reward = model[(
                    memory_prev_features, memory_action)]
                train_network(
                    q_network, target_network, optimizer,
                    memory_prev_features, memory_action, memory_reward, memory_features,
                )

            prev_observation = observation

            if done:
                print(f'Episode {i + 1} finished after {t + 1} timesteps')
                break


def monte_carlo(
        env,
        state_converter,
        max_steps=200,
        first_visit=False,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    q_value = state_converter.get_initial_q_value()
    appearances = np.zeros_like(q_value)

    def update_q_value(q_value, state, action, G):
        index = state_converter.state_to_index(state)

        q_value[index][action] = \
            (q_value[index][action] * appearances[index][action] + G) / (appearances[index][action] + 1)
        appearances[index][action] += 1

    time_steps = []
    for i in range(episode_count):
        quadruplet = []
        prev_state = env.reset()
        seen_in_episode = set()
        for t in range(max_steps):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action * 2)
            reward = get_reward(prev_state, state)
            is_terminated = t + 1 == max_steps

            idx = state_converter.state_to_index(state)
            quadruplet.append((prev_state, action, reward, (idx, action) not in seen_in_episode))
            seen_in_episode.add((idx, action))

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        state_converter.next()
        G = 0
        for state, action, reward, first_flag in quadruplet[::-1]:
            G = G * gamma + reward
            if first_visit and first_flag:
                update_q_value(q_value, state, action, G)

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def sarsa(
        env,
        state_converter,
        max_steps=200,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    gamma_n = gamma ** n_step
    def update_q_value(q_value, prev_state, prev_action, state, action, return_value):
        prev_index = state_converter.state_to_index(prev_state)
        delta = return_value - q_value[prev_index][prev_action]
        if state is not None:
            index = state_converter.state_to_index(state)
            delta += gamma_n * q_value[index][action]
        q_value[prev_index][prev_action] += alpha * delta

    q_value = state_converter.get_initial_q_value()

    queue = Queue()

    time_steps = []
    for i in range(episode_count):
        if not queue.empty():
            queue.get_nowait()
        for _ in range(n_step):
            queue.put((None, None, 0))
        queue.get()

        G = 0
        prev_state = env.reset()
        prev_action = get_action(q_value, state_converter, prev_state)
        for t in range(max_steps):
            if render:
                env.render()

            state, reward, is_terminated, _ = env.step(prev_action)
            reward = -100 if is_terminated and t < int(max_steps * 0.975) else 1
            action = get_action(q_value, state_converter, state)

            queue.put((prev_state, prev_action, reward))
            target_state, target_action, target_reward = queue.get()
            G += reward * gamma_n
            G /= gamma

            if t + 1 >= n_step and target_state is not None:
                update_q_value(q_value, target_state, target_action, state, action, G)
            G -= target_reward

            prev_state = state
            prev_action = action
            if is_terminated:
                while not queue.empty():
                    target_state, target_action, target_reward = queue.get()
                    G /= gamma
                    update_q_value(q_value, target_state, target_action, None, None, G)
                    G -= target_reward

                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def q_learning(
        env,
        state_converter,
        max_steps=200,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    gamma_n = gamma ** n_step
    def update_q_value(q_value, prev_state, action, state, return_value):
        prev_index = state_converter.state_to_index(prev_state)
        delta = return_value - q_value[prev_index][action]
        if state is not None:
            index = state_converter.state_to_index(state)
            delta += gamma_n * np.max(q_value[index])
        q_value[prev_index][action] += alpha * delta

    q_value = state_converter.get_initial_q_value()

    queue = Queue()

    time_steps = []
    for i in range(episode_count):
        if not queue.empty():
            queue.get_nowait()
        for _ in range(n_step):
            queue.put((None, None, 0))
        queue.get()

        G = 0
        prev_state = env.reset()
        for t in range(max_steps):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)
            reward = get_reward(prev_state, state)

            queue.put((prev_state, action, reward))
            target_state, target_action, target_reward = queue.get()
            G += reward * gamma_n
            G /= gamma

            if t + 1 >= n_step and target_state is not None:
                update_q_value(q_value, target_state, target_action, state, G)
            G -= target_reward

            prev_state = state
            if is_terminated:
                while not queue.empty():
                    target_state, target_action, target_reward = queue.get()
                    G /= gamma
                    update_q_value(q_value, target_state, target_action, None, G)
                    G -= target_reward

                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def dyna_q(
        env,
        state_converter,
        max_steps=200,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    def update_q_value(q_value, prev_state_index, action, state_index, reward):
        delta = reward + gamma * np.max(q_value[state_index]) - q_value[prev_state_index][action]
        q_value[prev_state_index][action] += alpha * delta

    q_value = state_converter.get_initial_q_value()
    model = dict()
    memory = set()

    time_steps = []
    for i in range(episode_count):
        prev_state = env.reset()
        for t in range(max_steps):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)
            reward = -100 if is_terminated and t < int(max_steps * 0.975) else 1

            prev_state_index = state_converter.state_to_index(prev_state)
            state_index = state_converter.state_to_index(state)

            update_q_value(q_value, prev_state_index, action, state_index, reward)

            model[(prev_state_index, action)] = (state_index, reward)
            memory.add((prev_state_index, action))

            for memory_prev_state_index, memory_action in random.sample(memory, min(n_step, len(memory))):
                memory_state_index, memory_reward = model[(memory_prev_state_index, memory_action)]
                update_q_value(
                    q_value, memory_prev_state_index, memory_action, memory_state_index, memory_reward)

            prev_state = state
            if is_terminated:
                print(f'Episode {i} finished after {t + 1} timesteps')
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def prioritized_sweeping(
        env,
        state_converter,
        max_steps=200,
        n_step=1,
        alpha=0.1,
        gamma=0.95,
        thre=1.0,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):

    q_value = state_converter.get_initial_q_value()
    model = dict()
    state_in_retro = dict()
    p_queue = PriorityQueue()

    time_steps = []
    for i in range(episode_count):
        prev_state = env.reset()
        for t in range(max_steps):
            if render:
                env.render()

            action = get_action(q_value, state_converter, prev_state)
            state, reward, is_terminated, _ = env.step(action)

            prev_state_index = state_converter.state_to_index(prev_state)
            state_index = state_converter.state_to_index(state)

            delta = reward + gamma * np.max(q_value[state_index]) - q_value[prev_state_index][action]
            q_value[prev_state_index][action] += alpha * delta
            if -abs(delta) > thre:
                p_queue.put((-abs(delta), prev_state_index, action))
            model[(prev_state_index, action)] = (state_index, reward)
            if state_index not in state_in_retro:
                state_in_retro[state_index] = set()
            state_in_retro[state_index].add((prev_state_index, action, reward))

            for _ in range(n_step):
                if p_queue.empty():
                    break
                _, memory_prev_state_index, memory_action = p_queue.get()
                memory_state_index, memory_reward = model[(memory_prev_state_index, memory_action)]
                delta = reward + gamma * np.max(q_value[state_index]) - q_value[prev_state_index][action]
                q_value[prev_state_index][action] += alpha * delta

                if memory_prev_state_index in state_in_retro:
                    for s_bar, a_bar, r_bar in state_in_retro[memory_prev_state_index]:
                        delta = r_bar + gamma * np.max(q_value[memory_prev_state_index]) - q_value[s_bar][a_bar]
                        if -abs(delta) > thre:
                            p_queue.put((-abs(delta), s_bar, a_bar))

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def sarsa_lambda(
        env,
        state_converter,
        max_steps=200,
        alpha=0.1,
        gamma=0.95,
        lambda_value=0.2,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    alpha *= 1 - lambda_value
    q_value = state_converter.get_initial_q_value()

    time_steps = []
    aaa = np.ones(8) * -1000
    bbb = np.ones(8) * 1000
    for i in range(episode_count):
        z = np.zeros_like(q_value)
        prev_state = env.reset()
        prev_action = get_action(q_value, state_converter, prev_state)
        q_value_old = 0
        for t in range(max_steps):
            if render:
                env.render()

            state, reward, is_terminated, _ = env.step(prev_action)
            for j in range(8):
                aaa[j] = max(aaa[j], state[j])
                bbb[j] = min(bbb[j], state[j])
            if abs(reward) > 5.0:
                print(reward)
            action = get_action(q_value, state_converter, state)

            if prev_action is not None:
                prev_index = state_converter.state_to_index(prev_state)
                index = state_converter.state_to_index(state)

                delta = reward + gamma * q_value[index][action] - q_value[prev_index][prev_action]

                x = np.zeros_like(q_value)
                x[prev_index][prev_action] = 1
                z = gamma * lambda_value * z + (1 - alpha * gamma * lambda_value * z[prev_index][prev_action]) * x

                q_value += alpha * (delta + q_value[prev_index][prev_action] - q_value_old) * z
                q_value -= alpha * (q_value[prev_index][prev_action] - q_value_old) * x
                q_value_old = q_value[index][action]

            prev_state = state
            prev_action = action
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

        if (i + 1) % 20 == 0:
            plot(time_steps, 'Time Step', figure_path)

def run():
    set_logger()
    args = get_args()
    set_seed(args.seed)

    figure_path = os.path.join(args.output_dir, args.time_step_plot)
    setup_output_dir(args.output_dir, dict(args._get_kwargs())) #pylint: disable=protected-access

    state_converter = LunarLanderStateConverter(epsilon=args.epsilon, episode_count=args.episode_count)

    env = gym.make('LunarLander-v2')
    if args.algorithm == 'dqn':
        dqn(
            env,
            render=args.render,
        )
    if args.algorithm == 'montecarlo':
        monte_carlo(
            env,
            state_converter,
            max_steps=args.max_steps,
            first_visit=args.first_visit,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'sarsa':
        sarsa(
            env,
            state_converter,
            max_steps=args.max_steps,
            n_step=args.n_step,
            alpha=args.alpha,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'qlearning':
        q_learning(
            env,
            state_converter,
            max_steps=args.max_steps,
            n_step=args.n_step,
            alpha=args.alpha,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'dynaq':
        dyna_q(
            env,
            state_converter,
            max_steps=args.max_steps,
            n_step=args.n_step,
            alpha=args.alpha,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'prioritized':
        prioritized_sweeping(
            env,
            state_converter,
            max_steps=args.max_steps,
            n_step=args.n_step,
            alpha=args.alpha,
            gamma=args.gamma,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    elif args.algorithm == 'sarsalambda':
        sarsa_lambda(
            env,
            state_converter,
            max_steps=args.max_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            lambda_value=args.lambda_value,
            episode_count=args.episode_count,
            render=args.render,
            figure_path=figure_path,
        )
    env.close()

if __name__ == '__main__':
    run()
