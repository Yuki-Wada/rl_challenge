import copy
import time
import numpy as np

import torch

from examples.network import SLPolicy, RolloutPolicy, Value


class RandomPolicy:
    def __init__(self, seed):
        from gym.utils import seeding
        np_random, _ = seeding.np_random(seed)
        self.np_random = np_random

    def get_move(self, state, env, color=0):
        possible_places = env.get_possible_actions(state, color)
        # No places left
        if len(possible_places) == 0:
            return d ** 2 + 1
        a = self.np_random.randint(len(possible_places))

        return possible_places[a]

    def get_action_probs(self, state, env):
        possible_places = env.get_possible_actions(state, 0)
        # No places left
        if len(possible_places) == 0:
            return d ** 2 + 1
        a = self.np_random.randint(len(possible_places))
        return [[a, 1 / len(possible_places)] for a in possible_places]

    def update_with_move(self, last_move):
        pass

class Node(object):
    def __init__(self, parent=None, prob=0):
        self.parent = parent
        self.children = {}  # Dictionary of Node with key:action
        self.n_visits = 0
        self.Q = 0
        # This value for u will be overwritten in the first call to update()
        self.u = prob + 0.1
        self.P = prob + 0.1

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) < 1

    def expand(self, action_probs):
        """Expand tree by creating new children.
        Arguments:
        action_probs -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.
        Returns:
        None
        """
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
        Returns:
        A tuple of (action, next_node)
        """
        for k in self.children:
            self.children[k].u = self.children[k].U(c_puct)
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value())

    def U(self, c_puct):
        return c_puct * self.P * np.sqrt(self.parent.n_visits) / (0.01 + self.n_visits)

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        Returns:
        None
        """
        # Count visit.
        self.n_visits += 1
        # Update Q, a running average of values for all visits.
        self.Q += (leaf_value - self.Q) / self.n_visits
        # Update u, the prior weighted by an exploration hyperparameter c_puct
        # Note that u is not normalized to be a distribution.

    def update_recursive(self, leaf_value, c_puct):
        self.update(leaf_value, c_puct)
        # If it is not root, this node's parent should be updated next.
        if not self.is_root():
            self.parent.update_recursive(leaf_value, c_puct)

    def get_value(self):
        return self.Q + self.u

    def copy(self):
        return copy.deepcopy(self)


class MCTS:
    def __init__(self, lmbda=0.5, c_puct=1, n_thr=15, time_limit=1):
        self.root = Node(None, 1.0)

        self.policy_net = SLPolicy()
        self.policy_net.load_state_dict(torch.load('data/model/sl_model.pth'))
        self.policy_net.eval()

        self.value_net = Value()
        self.value_net.load_state_dict(torch.load('data/model/value_model.pth'))
        self.value_net.eval()

        self.rollout_net = RolloutPolicy()
        self.rollout_net.load_state_dict(torch.load('data/model/rollout_model.pth'))
        self.rollout_net.eval()

        self.lmbda = lmbda
        self.c_puct = c_puct
        self.n_thr = n_thr
        self.time_limit = time_limit

    def make_state_var(self, state, color):
        state = state[0] + state[1] * 2
        if color == 0:
            tmp = 3 * np.ones([8,8], dtype=np.float32)
            state = state * (tmp - state) * (tmp - state) / 2
        state_2ch = np.stack([state==1, state==2], axis=0).astype(np.float32)
        state_var = torch.Tensor(state_2ch.reshape(2,1,8,8).transpose(1,0,2,3))
        return state_var

    def policy_func(self, state, color, actions, env):
        state_var = self.make_state_var(state, color)
        prob = self.policy_net(state_var).data.reshape(64)
        action_probs = []
        for action in actions:
            action_probs.append((action, prob[action]))
        return action_probs

    def value_func(self, state, color):
        state_var = self.make_state_var(state, color)
        return self.value_net(state_var).data.reshape(1)[0]

    def playout(self, state, color, node, env):
        if node.is_leaf():
            if node.n_visits >= self.n_thr:
            # a list of tuples of actions and their prior probability
                actions = env.get_possible_actions(state, color)
                if len(actions) < 1:
                    # Pass
                    node.children[-1] = Node(node, 1)
                if len(actions) == 1:
                    # Have only one choice
                    node.children[actions[0]] = Node(node, 1)
                else:
                    action_probs = self.policy_func(state, color, actions, env)
                    node.expand(action_probs)
                self.playout(state, color, node, env)
            else:
                v = self.value_func(state, color) if self.lmbda < 1 else 0
                z = self.evaluate_rollout(state, color, copy.deepcopy(env)) if self.lmbda > 0 else 0
                leaf_value = (1 - self.lmbda) * v + self.lmbda * z
                # Update value and visit count of nodes in this traversal.
                node.update_recursive(leaf_value, self.c_puct)

        else:
            action, node = node.select(self.c_puct)
            state, _, _, _ = env.step(action)
            color = 1 - color
            self.playout(state, color, node, env)

    def evaluate_rollout(self, state, color, env):
        reward = 0
        done = False
        while not done:
            state_var = self.make_state_var(state, color)
            prob = self.rollout_net(state_var).data.numpy().reshape(64)

            valid = np.zeros(64)
            actions = []
            for action in env.get_possible_actions(state, color):
                if action < 64:
                    actions.append(action)
                if action == 65:
                    return reward

            valid[actions] = 1
            prob = prob * valid

            action = np.random.choice(64, p=prob/np.sum(prob))
            state, reward, done, _ = env.step(action, color)

            color = 1 - color

        return reward * (1 if color == 0 else -1)

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

    def get_move(self, state, env, color):
        start = time.time()
        elapsed = 0
        while not self.root.children.items() or elapsed < self.time_limit:
            state_copy = state.copy()
            self.playout(state_copy, color, self.root, copy.deepcopy(env))
            elapsed = time.time() - start

        # chosen action is the *most visited child*, not the highest-value
        action = max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]
        self.update_with_move(action)

        return action
