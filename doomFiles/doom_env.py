import os
import multiprocessing

import logging

import numpy as np

import gym
from gym import spaces, error
from gym.utils import seeding
from time import sleep

import vizdoom
from vizdoom import DoomGame, Mode, Button, GameVariable, ScreenFormat, ScreenResolution, Loader, doom_fixed_to_double

logger = logging.getLogger(__name__)

# Arguments:
RANDOMIZE_MAPS = 80  # 0 means load default, otherwise randomly load in the id mentioned
NO_MONSTERS = True  # remove monster spawning

# Constants
NUM_ACTIONS = 43
NUM_LEVELS = 9
CONFIG = 0
SCENARIO = 1
MAP = 2
DIFFICULTY = 3
ACTIONS = 4
MIN_SCORE = 5
TARGET_SCORE = 6

# Format (config, scenario, map, difficulty, actions, min, target)
DOOM_SETTINGS = [
    ['basic.cfg', 'basic.wad', 'map01', 5, [0, 10, 11], -485, 10],                               # 0 - Basic
    ['deadly_corridor.cfg', 'deadly_corridor.wad', '', 1, [0, 10, 11, 13, 14, 15], -120, 1000],  # 1 - Corridor
    ['defend_the_center.cfg', 'defend_the_center.wad', '', 5, [0, 14, 15], -1, 10],              # 2 - DefendCenter
    ['defend_the_line.cfg', 'defend_the_line.wad', '', 5, [0, 14, 15], -1, 15],                  # 3 - DefendLine
    ['health_gathering.cfg', 'health_gathering.wad', 'map01', 5, [13, 14, 15], 0, 1000],         # 4 - HealthGathering
    ['my_way_home.cfg', 'my_way_home_dense.wad', '', 5, [13, 14, 15], -0.22, 0.5],                     # 5 - MyWayHome
    ['predict_position.cfg', 'predict_position.wad', 'map01', 3, [0, 14, 15], -0.075, 0.5],      # 6 - PredictPosition
    ['take_cover.cfg', 'take_cover.wad', 'map01', 5, [10, 11], 0, 750],                          # 7 - TakeCover
    ['deathmatch.cfg', 'deathmatch.wad', '', 5, [x for x in range(NUM_ACTIONS) if x != 33], 0, 20], # 8 - Deathmatch
    ['my_way_home.cfg', 'my_way_home_sparse.wad', '', 5, [13, 14, 15], -0.22, 0.5],               # 9 - MyWayHomeFixed
    ['my_way_home.cfg', 'my_way_home_verySparse.wad', '', 5, [13, 14, 15], -0.22, 0.5],               # 10 - MyWayHomeFixed15
]


# Singleton pattern
class DoomLock:
    class __DoomLock:
        def __init__(self):
            self.lock = multiprocessing.Lock()
    instance = None
    def __init__(self):
        if not DoomLock.instance:
            DoomLock.instance = DoomLock.__DoomLock()
    def get_lock(self):
        return DoomLock.instance.lock



class DoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 35}

    def __init__(self, level):
        self.previous_level = -1
        self.level = level
        self.game = DoomGame()
        self.loader = Loader()
        self.doom_dir = os.path.dirname(os.path.abspath(__file__))
        self._mode = 'algo'  # 'algo' or 'human'
        self.no_render = False  # To disable double rendering in human mode
        self.viewer = None
        self.is_initialized = False  # Indicates that reset() has been called
        self.curr_seed = 0
        self.lock = (DoomLock()).get_lock()
        self.action_space = spaces.MultiDiscrete([[0, 1]] * 38 + [[-10, 10]] * 2 + [[-100, 100]] * 3)
        self.allowed_actions = list(range(NUM_ACTIONS))
        self.screen_height = 480
        self.screen_width = 640
        self.screen_resolution = ScreenResolution.RES_640X480
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self._seed()
        self._configure()

    def _configure(self, lock=None, **kwargs):
        if 'screen_resolution' in kwargs:
            logger.warn(
                'Deprecated - Screen resolution must now be set using a wrapper. See documentation for details.')
        # Multiprocessing lock
        if lock is not None:
            self.lock = lock

    def _start_episode(self):
        if self.curr_seed > 0:
            self.game.set_seed(self.curr_seed)
            self.curr_seed = 0
        self.game.new_episode()
        return

    def _play_human_mode(self):
        while not self.game.is_episode_finished():
            self.game.advance_action()
            state = self.game.get_state()
            total_reward = self.game.get_total_reward()
            info = self._get_game_variables(state.game_variables)
            info["TOTAL_REWARD"] = round(total_reward, 4)
            print('===============================')
            print('State: #' + str(state.number))
            print('Action: \t' + str(self.game.get_last_action()) + '\t (=> only allowed actions)')
            print('Reward: \t' + str(self.game.get_last_reward()))
            print('Total Reward: \t' + str(total_reward))
            print('Variables: \n' + str(info))
            sleep(0.02857)  # 35 fps = 0.02857 sleep between frames
        print('===============================')
        print('Done')
        return


