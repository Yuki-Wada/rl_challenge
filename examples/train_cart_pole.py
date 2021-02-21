"""
Train a cart pole task.
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
from queue import Queue, PriorityQueue
import heapq
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mltools.utils import set_seed, set_logger, dump_json

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

class CartPoleStateConverter:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

        self.min_position = -2.4
        self.max_position = 2.4
        self.position_size = 6

        self.min_velocity = -2.0
        self.max_velocity = 2.0
        self.velocity_size = 6

        self.min_angle = -0.5
        self.max_angle = 0.5
        self.angle_size = 6

        self.min_angular_velocity = -1.5
        self.max_angular_velocity = 1.5
        self.angular_velocity_size = 6

        self.action_num = 2

    def state_to_index(self, state):
        def bins(clip_min, clip_max, num):
            return np.linspace(clip_min, clip_max, num + 1)[1:-1]

        position, velocity, angle, angular_velocity = state

        position_index = np.digitize(position, bins=bins(
            self.min_position, self.max_position, self.position_size))
        velocity_index = np.digitize(velocity, bins=bins(
            self.min_velocity, self.max_velocity, self.velocity_size))
        angle_index = np.digitize(angle, bins=bins(
            self.min_angle, self.max_angle, self.angle_size))
        angular_velocity_index = np.digitize(angular_velocity, bins=bins(
            self.min_angular_velocity, self.max_angular_velocity, self.angular_velocity_size))
        
        return position_index, velocity_index, angle_index, angular_velocity_index

    def get_initial_q_value(self):
        q_value = np.random.uniform(low=0, high=1, size=(
            self.position_size,
            self.velocity_size,
            self.angle_size,
            self.angular_velocity_size,
            self.action_num,
        ))
        q_value[0] = 0
        q_value[-1] = 0
        q_value[:, :, 0] = 0
        q_value[:, :, -1] = 0

        return q_value

def get_action(q_value, state_converter, state):
    index = state_converter.state_to_index(state)
    if np.random.uniform() <= state_converter.epsilon:
        return np.random.randint(0, state_converter.action_num)
    return np.argmax(q_value[index])

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
            state, reward, is_terminated, _ = env.step(action)
            reward = -10 if is_terminated and t < int(max_steps * 0.975) else 1

            idx = state_converter.state_to_index(state)
            quadruplet.append((prev_state, action, reward, (idx, action) not in seen_in_episode))
            seen_in_episode.add((idx, action))

            prev_state = state
            if is_terminated:
                print("Episode {} finished after {} timesteps".format(i, t + 1))
                time_steps.append(t + 1)
                break

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
            reward = -100 if is_terminated and t < int(max_steps * 0.975) else 1 

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
            reward = -100 if is_terminated and t < int(max_steps * 0.975) else 1 

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
    for i in range(episode_count):
        z = np.zeros_like(q_value)
        prev_state = env.reset()
        prev_action = get_action(q_value, state_converter, prev_state)
        q_value_old = 0
        for t in range(max_steps):
            if render:
                env.render()

            state, reward, is_terminated, _ = env.step(prev_action)
            reward = -100 if is_terminated and t < int(max_steps * 0.975) else 1
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

    state_converter = CartPoleStateConverter(epsilon=args.epsilon)

    gym.envs.registration.register(
        id='CartPole-v2',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=args.max_steps,
        reward_threshold=int(args.max_steps * 0.975),
    )

    env = gym.make('CartPole-v2')
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
