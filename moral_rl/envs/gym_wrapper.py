import envs.randomized_v1
import envs.randomized_v2 
import envs.randomized_v3
import envs.randomized_v1_test
import envs.randomized_v4
import envs.randomized_v3_test
import envs.randomized_v3_test_discrim_eval
from pycolab import rendering
from typing import Callable
import gym
from gym import spaces
from gym.utils import seeding
import copy
import numpy as np
import time

from stable_baselines3.common.utils import set_random_seed


class GymWrapper(gym.Env):
    """Gym wrapper for pycolab environment"""

    def __init__(self, env_id):
        self.env_id = env_id

        if env_id == 'randomized_v2':
            self.layers = ('#', 'P', 'C', 'H', 'G')
            self.width = 8
            self.height = 8
            self.num_actions = 9
        elif env_id == 'randomized_v3':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
        elif env_id == 'randomized_v1':
            self.layers = ('#', 'P', 'F', 'C')
            self.width = 5
            self.height = 5
            self.num_actions = 9
        elif env_id == 'randomized_v1_test':
            self.layers = ('#', 'P', 'F', 'C')
            self.width = 5
            self.height = 5
            self.num_actions = 9
        elif env_id == 'randomized_v4':
            self.layers = ('#', 'P', 'F', 'C')
            self.width = 8
            self.height = 8
            self.num_actions = 9
        elif env_id == 'randomized_v3_test':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
        elif env_id == 'randomized_v3_test_discrim_eval':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
            

        self.game = None
        self.np_random = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height, len(self.layers)),
            dtype=np.int32
        )

        self.renderer = rendering.ObservationToFeatureArray(self.layers)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _obs_to_np_array(self, obs):
        return copy.copy(self.renderer(obs))

    def reset(self):
        if self.env_id == 'randomized_v2':
            self.game = envs.randomized_v2.make_game()
        elif self.env_id == 'randomized_v3':
            self.game = envs.randomized_v3.make_game()
        elif self.env_id == 'randomized_v1':
            self.game = envs.randomized_v1.make_game()
        elif self.env_id == 'randomized_v1_test':
            self.game = envs.randomized_v1_test.make_game()
        elif self.env_id == 'randomized_v4':
            self.game = envs.randomized_v4.make_game()
        elif self.env_id == 'randomized_v3_test':
            self.game = envs.randomized_v3_test.make_game()
        elif self.env_id == 'randomized_v3_test_discrim_eval':
            self.game = envs.randomized_v3_test_discrim_eval.make_game()
            # self.game = envs.randomized_v3_test_discrim_eval.make_game(rand_reset=True, dico_pos=None)

        obs, _, _ = self.game.its_showtime()
        return self._obs_to_np_array(obs)

    def step(self, action):
        obs, reward, _ = self.game.play(action)
        return self._obs_to_np_array(obs), reward, self.game.game_over, self.game.the_plot

    def step_demo(self, action):
        obs, reward, discount = self.game.play(action)
        return obs, reward, discount, self._obs_to_np_array(obs), self.game.game_over, self.game.the_plot

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = GymWrapper(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

def make_env_eval_discrim(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = GymWrapper_eval_discrim(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

class GymWrapper_eval_discrim(GymWrapper):
    """Gym wrapper for pycolab environment"""

    def __init__(self, env_id):
        self.env_id = env_id

        if env_id == 'randomized_v3_test_discrim_eval':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
            

        self.game = None
        self.np_random = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height, len(self.layers)),
            dtype=np.int32
        )

        self.renderer = rendering.ObservationToFeatureArray(self.layers)

        self.seed()

        # we don't call reset here because the show_time fct will be called in ui.play(game)
        # self.reset()
        # we call this fct that initialize the game but don't call show_time
        self.reset_but_no_show_time_yet()

    def reset_but_no_show_time_yet(self):
        if self.env_id == 'randomized_v3_test_discrim_eval':
            self.game = envs.randomized_v3_test_discrim_eval.make_game()
            # self.game = envs.randomized_v3_test_discrim_eval.make_game(rand_reset=True, dico_pos=None)
            

        # obs, _, _ = self.game.its_showtime()
        # return self._obs_to_np_array(obs)
        return 0

    # def reset_with_defined_board_but_no_show_time_yet(self, the_plot):
    #     if self.env_id == 'randomized_v3_test_discrim_eval':
    #         self.game = envs.randomized_v3_test_discrim_eval.make_game_eval_discrim(the_plot)
    #         # self.game = envs.randomized_v3_test_discrim_eval.make_game(rand_reset=True, dico_pos=None)
            

    #     # obs, _, _ = self.game.its_showtime()
    #     # return self._obs_to_np_array(obs)
    #     return 0

    def reset_with_defined_board(self, the_plot):
        if self.env_id == 'randomized_v3_test_discrim_eval':
            self.game = envs.randomized_v3_test_discrim_eval.make_game_eval_discrim(the_plot)
            self.gamegame.the_plot['P_pos'] = the_plot['P_pos']
            self.gamegame.the_plot['F_pos'] = the_plot['F_pos']
            self.gamegame.the_plot['C_pos'] = the_plot['C_pos']
            self.gamegame.the_plot['S_pos'] = the_plot['S_pos']
            self.gamegame.the_plot['V_pos'] = the_plot['V_pos']
            # self.game = envs.randomized_v3_test_discrim_eval.make_game(rand_reset=True, dico_pos=None)
            

        # obs, _, _ = self.game.its_showtime()
        # return self._obs_to_np_array(obs)
        # return 0

    def step_demo(self, action):
        obs, reward, discount = self.game.play(action)
        return obs, reward, discount, self._obs_to_np_array(obs), self.game.game_over, self.game.the_plot


def make_env_demo(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = GymWrapper_demo(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init

class GymWrapper_demo(GymWrapper):
    """Gym wrapper for pycolab environment"""

    def __init__(self, env_id):
        self.env_id = env_id

        if env_id == 'randomized_v2':
            self.layers = ('#', 'P', 'C', 'H', 'G')
            self.width = 8
            self.height = 8
            self.num_actions = 9
        elif env_id == 'randomized_v3':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
        elif env_id == 'randomized_v1':
            self.layers = ('#', 'P', 'F', 'C')
            self.width = 5
            self.height = 5
            self.num_actions = 9
        elif env_id == 'randomized_v1_test':
            self.layers = ('#', 'P', 'F', 'C')
            self.width = 5
            self.height = 5
            self.num_actions = 9
        elif env_id == 'randomized_v4':
            self.layers = ('#', 'P', 'F', 'C')
            self.width = 8
            self.height = 8
            self.num_actions = 9
        elif env_id == 'randomized_v3_test':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
        elif env_id == 'randomized_v3_test_discrim_eval':
            self.layers = ('#', 'P', 'F', 'C', 'S', 'V')
            self.width = 16
            self.height = 16
            self.num_actions = 9
            

        self.game = None
        self.np_random = None

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.width, self.height, len(self.layers)),
            dtype=np.int32
        )

        self.renderer = rendering.ObservationToFeatureArray(self.layers)

        self.seed()

        # we don't call reset here because the show_time fct will be called in ui.play(game)
        # self.reset()
        # we call this fct that initialize the game but don't call show_time
        self.reset_but_no_show_time_yet()

    def reset_but_no_show_time_yet(self):
        if self.env_id == 'randomized_v2':
            self.game = envs.randomized_v2.make_game()
        elif self.env_id == 'randomized_v3':
            self.game = envs.randomized_v3.make_game()
        elif self.env_id == 'randomized_v1':
            self.game = envs.randomized_v1.make_game()
        elif self.env_id == 'randomized_v1_test':
            self.game = envs.randomized_v1_test.make_game()
        elif self.env_id == 'randomized_v4':
            self.game = envs.randomized_v4.make_game()
        elif self.env_id == 'randomized_v3_test':
            self.game = envs.randomized_v3_test.make_game()
        elif self.env_id == 'randomized_v3_test_discrim_eval':
            self.game = envs.randomized_v3_test_discrim_eval.make_game()
            # self.game = envs.randomized_v3_test_discrim_eval.make_game(rand_reset=True, dico_pos=None)
            

        # obs, _, _ = self.game.its_showtime()
        # return self._obs_to_np_array(obs)
        return 0

    def step(self, action):
        obs, reward, discount = self.game.play(action)
        return obs, reward, discount, self._obs_to_np_array(obs), self.game.game_over, self.game.the_plot

class VecEnv:
    def __init__(self, env_id, n_envs):
        self.env_list = [make_env(env_id, i, (int(str(time.time()).replace('.', '')[-8:]) + i))() for i in range(n_envs)]
        self.n_envs = n_envs
        self.env_id = env_id
        self.action_space = self.env_list[0].action_space
        self.observation_space = self.env_list[0].observation_space

    def reset(self):
        obs_list = []
        for i in range(self.n_envs):
            obs_list.append(self.env_list[i].reset())

        return np.stack(obs_list, axis=0)

    def step(self, actions):
        obs_list = []
        rew_list = []
        done_list = []
        info_list = []
        for i in range(self.n_envs):
            obs_i, rew_i, done_i, info_i = self.env_list[i].step(actions[i])

            if done_i:
                obs_i = self.env_list[i].reset()

            obs_list.append(obs_i)
            rew_list.append(rew_i)
            done_list.append(done_i)
            info_list.append(info_i)

        return np.stack(obs_list, axis=0), rew_list, done_list, info_list
