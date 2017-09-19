import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile
import random

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from baselines.deepq.simple import ActWrapper

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

import gflags as flags

# Minimap feature ids
_MMAP_HEIGHT_MAP = features.MINIMAP_FEATURES.height_map.index
_MMAP_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MMAP_CREEP = features.MINIMAP_FEATURES.creep.index
_MMAP_CAMERA = features.MINIMAP_FEATURES.camera.index
_MMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_MMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MMAP_SELECTED = features.MINIMAP_FEATURES.selected.index

# Screen feature ids
_HEIGHT_MAP = features.SCREEN_FEATURES.height_map.index
_VISIBILITY = features.SCREEN_FEATURES.visibility_map.index
_CREEP = features.SCREEN_FEATURES.creep.index
_POWER = features.SCREEN_FEATURES.power.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_HP = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_EN = features.SCREEN_FEATURES.unit_energy.index
_UNIT_SH = features.SCREEN_FEATURES.unit_shields.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index

# For indexing into player relative features
_PLAYER_BACKGROUND = 0
_PLAYER_SELF = 1
_PLAYER_ALLY = 2
_PLAYER_NEUTRAL = 3
_PLAYER_HOSTILE = 4

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

_SCREEN = [0]

_SELECT_POINT_SELECT = 0
_SELECT_POINT_TOGGLE = 1

_TERRAN_MARINE = 48

_MIN_RES = 0
_MAX_RES = 63
_MOVE_STEP_SIZE = 4

FLAGS = flags.FLAGS


class MarineSquad:
    def __init__(self):
        self.select_unit_loc = (0, 0)

    def select_action(self, obs, action_id):
        try:
            if action_id == 0:
                action = self.select_high_hp_marine(obs)
            elif action_id == 1:
                action = self.select_low_hp_marine(obs)
            elif action_id == 2:
                action = self.select_random_marine(obs)
            elif action_id == 3:
                action = self.select_army(obs)
            elif action_id == 4:
                action = self.move_up(obs)
            elif action_id == 5:
                action = self.move_down(obs)
            elif action_id == 6:
                action = self.move_left(obs)
            elif action_id == 7:
                action = self.move_right(obs)
            elif action_id == 8:
                action = self.attack_up(obs)
            elif action_id == 9:
                action = self.attack_down(obs)
            elif action_id == 10:
                action = self.attack_left(obs)
            elif action_id == 11:
                action = self.attack_right(obs)
            elif action_id == 12:
                action = self.attack_lowest_hp(obs)
            elif action_id == 13:
                action = sc2_actions.FunctionCall(_NO_OP, [])
        except Exception as e:
            # print(e)
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return [action]

    def select_high_hp_marine(self, obs):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_hp = obs.observation["screen"][_UNIT_HP]

        friendly_units = (player_relative == _PLAYER_SELF)
        marines = (unit_type == _TERRAN_MARINE)

        # Locations of your own marines relative to screen
        friendly_marines_y, friendly_marines_x = np.logical_and(friendly_units, marines).nonzero()

        marine_ind = np.argmax(unit_hp[friendly_marines_y, friendly_marines_x])

        self.select_unit_loc = (friendly_marines_x[marine_ind], friendly_marines_y[marine_ind])

        action = sc2_actions.FunctionCall(_SELECT_POINT,
                                          [_SCREEN, self.select_unit_loc])

        return action

    def select_low_hp_marine(self, obs):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_hp = obs.observation["screen"][_UNIT_HP]

        friendly_units = (player_relative == _PLAYER_SELF)
        marines = (unit_type == _TERRAN_MARINE)

        # Locations of your own marines relative to screen
        friendly_marines_y, friendly_marines_x = np.logical_and(friendly_units, marines).nonzero()

        marine_ind = np.argmin(unit_hp[friendly_marines_y, friendly_marines_x])

        self.select_unit_loc = (friendly_marines_x[marine_ind], friendly_marines_y[marine_ind])

        action = sc2_actions.FunctionCall(_SELECT_POINT,
                                          [_SCREEN, self.select_unit_loc])

        return action

    def select_random_marine(self, obs):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_type = obs.observation["screen"][_UNIT_TYPE]

        friendly_units = (player_relative == _PLAYER_SELF)
        marines = (unit_type == _TERRAN_MARINE)

        # Locations of your own marines relative to screen
        friendly_marines_y, friendly_marines_x = np.logical_and(friendly_units, marines).nonzero()

        marine_ind = np.random.choice(np.arange(len(friendly_marines_y)),1)
        self.select_unit_loc = (friendly_marines_x[marine_ind], friendly_marines_y[marine_ind])

        action = sc2_actions.FunctionCall(_SELECT_POINT,
                                          [_SCREEN, self.select_unit_loc])

        return action

    def select_army(self, obs):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        friendly_units_y, friendly_units_x = (player_relative == _PLAYER_SELF).nonzero()
        self.select_unit_loc = (int(friendly_units_x.mean()), int(friendly_units_y.mean()))

        action = sc2_actions.FunctionCall(_SELECT_ARMY, [[0]])

        return action

    def move_up(self, obs):
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            y = max(_MIN_RES, y - int(_MOVE_STEP_SIZE))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def move_down(self, obs):
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            y = min(_MAX_RES, y + int(_MOVE_STEP_SIZE))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def move_left(self, obs):
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            x = max(_MIN_RES, x - int(_MOVE_STEP_SIZE))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def move_right(self, obs):
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            x = min(_MAX_RES, x + int(_MOVE_STEP_SIZE))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def attack_up(self, obs):
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            y = max(_MIN_RES, y - int(_MOVE_SCREEN))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def attack_down(self, obs):
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            y = min(_MAX_RES, y + int(_MOVE_SCREEN))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def attack_left(self, obs):
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            x = max(_MIN_RES, x - int(_MOVE_SCREEN))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def attack_right(self, obs):
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            x, y = self.select_unit_loc
            x = min(_MAX_RES, x + int(_MOVE_SCREEN))
            self.select_unit_loc = (x, y)
            action = sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, (x, y)])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def attack_lowest_hp(self, obs):
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            unit_hp = obs.observation["screen"][_UNIT_HP]
            enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()
            enemy_ind = np.argmin(unit_hp[enemy_y, enemy_x])
            target = (enemy_x[enemy_ind], enemy_y[enemy_ind])
            action = sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action

    def attack_closest_enemy(self, obs):
        if _ATTACK_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            enemy_pos = (player_relative == _PLAYER_HOSTILE).nonzero()
            unit_pos = np.array(self.select_unit_loc)
            target = enemy_pos[np.argmin(np.linalg.norm(enemy_pos - unit_pos))]
            action = sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, target])
        else:
            action = sc2_actions.FunctionCall(_NO_OP, [])

        return action



