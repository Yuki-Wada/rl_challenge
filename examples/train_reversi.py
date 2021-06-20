"""
Train Reversi.
"""
import os
import argparse
import logging
import time
import numpy as np
import gym
import pyglet

from examples.utils import set_seed, set_logger, dump_json
from examples.reversi_env import ReversiEnv
from examples.mcts import RandomPolicy, MCTS

logger = logging.getLogger(__name__)


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

class Drawer:
    GRID_SIZE = 30
    ROW_NUM = 8
    COL_NUM = 8
    MAP_COLORS = {
        'B': (255, 255, 255),
        'W': (  0,   0,   0),
        'O': ( 64, 192,   0),
    }

    def __init__(self):
        self._window = pyglet.window.Window(Drawer.GRID_SIZE * Drawer.ROW_NUM, Drawer.GRID_SIZE * Drawer.COL_NUM)
        self._window.set_caption("Reversi")

    def draw_lines(self):
        for pos_x in range(Drawer.COL_NUM):
            line = pyglet.graphics.vertex_list(
                2,
                ('v2f', [pos_x * Drawer.GRID_SIZE, 0, pos_x * Drawer.GRID_SIZE, Drawer.GRID_SIZE * Drawer.ROW_NUM]),
                ('c3B', [0, 0, 0, 0, 0, 0]),
            )
            line.draw(pyglet.gl.GL_LINES)
        for pos_y in range(Drawer.ROW_NUM):
            line = pyglet.graphics.vertex_list(
                2,
                ('v2f', [0, pos_y * Drawer.GRID_SIZE, Drawer.GRID_SIZE * Drawer.ROW_NUM, pos_y * Drawer.GRID_SIZE]),
                ('c3B', [0, 0, 0, 0, 0, 0]),
            )
            line.draw(pyglet.gl.GL_LINES)

    def draw_rect_angle(self, x, y, width, height, color):
        rect_angle = pyglet.graphics.vertex_list(
            4,
            ('v2f', [x, y, x + width, y, x + width, y + height, x, y + height]),
            ('c3B', sum([color for _ in range(4)], tuple())),
        )
        rect_angle.draw(pyglet.gl.GL_QUADS)

    def draw_circle(self, x, y, radius, color):
        points = 20
        vertex = []
        for i in range(points):
            angle = 2 * np.pi * i / points
            vertex += [x + radius * np.cos(angle), y + + radius * np.sin(angle)]

        circle = pyglet.graphics.vertex_list(
            points, ('v2f', vertex), ('c3B', sum([color for _ in range(points)], tuple())),
        )
        circle.draw(pyglet.gl.GL_POLYGON)

    def draw(self, state):
        self._window.clear()
        for pos_y in range(Drawer.ROW_NUM):
            for pos_x in range(Drawer.COL_NUM):
                color = Drawer.MAP_COLORS['O']
                self.draw_rect_angle(
                    x=pos_x * Drawer.GRID_SIZE,
                    y=(Drawer.ROW_NUM - pos_y - 1) * Drawer.GRID_SIZE,
                    width=Drawer.GRID_SIZE,
                    height=Drawer.GRID_SIZE,
                    color=color,
                )
        self.draw_lines()

        for pos_y in range(Drawer.ROW_NUM):
            for pos_x in range(Drawer.COL_NUM):
                mass_state_index = np.argmax(state[:, pos_y, pos_x])
                if mass_state_index != 2:
                    color_key = 'B' if mass_state_index == 1 else 'W'
                    color = Drawer.MAP_COLORS[color_key]
                    self.draw_circle(
                        x=(pos_x + 0.5) * Drawer.GRID_SIZE,
                        y=(Drawer.ROW_NUM - pos_y - 0.5) * Drawer.GRID_SIZE,
                        radius=Drawer.GRID_SIZE // 2 - 1,
                        color=color,
                    )

        self._tick()

    def _tick(self):
        pyglet.clock.tick()
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event("on_draw")
            window.flip()

def train_mcts(
        env,
        max_steps=200,
        alpha=0.1,
        gamma=0.95,
        lambda_value=0.2,
        episode_count=2000,
        render=False,
        figure_path=None,
    ):
    drawer = Drawer()
    # my_policy = RandomPolicy(0)
    my_policy = MCTS(time_limit=3)
    # opponent_policy = RandomPolicy(0)
    opponent_policy = MCTS(time_limit=1)

    for i_episode in range(episode_count):
        observation = env.reset()
        if render:
            drawer.draw(observation)
        for t in range(100):
            my_action = my_policy.get_move(observation, env, 0)
            my_policy.update_with_move(my_action)
            opponent_policy.update_with_move(my_action)

            observation, reward, done, info = env.step(my_action, 0)
            if render:
                if isinstance(my_policy, RandomPolicy):
                    time.sleep(1)
                drawer.draw(observation)

            opponent_action = opponent_policy.get_move(observation, env, 1)
            my_policy.update_with_move(opponent_action)
            opponent_policy.update_with_move(opponent_action)

            observation, reward, done, info = env.step(opponent_action, 1)
            if render:
                if isinstance(opponent_policy, RandomPolicy):
                    time.sleep(1)
                drawer.draw(observation)

            if done:
                black_score = len(np.where(env.state[0,:,:]==1)[0])
                print("Episode finished after {} timesteps".format(t+1))
                print(black_score)
                break

def run():
    env = ReversiEnv(
        player_color='black',
        opponent='random',
        observation_type='numpy3c',
        illegal_place_mode='lose',
        board_size=8,
    )
    env.reset()

    set_logger()
    args = get_args()
    set_seed(args.seed)

    train_mcts(
        env,
        episode_count=args.episode_count,
        render=args.render,
    )
    env.close()

if __name__ == '__main__':
    run()
