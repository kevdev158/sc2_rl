import numpy as np
import os, sys
import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from baselines.deepq.simple import ActWrapper

from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

from marine_squad import MarineSquad
import my_utils

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
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

_SUPPLY_USED = 3

FLAGS = flags.FLAGS

_USED_SCREEN_FEATURES = [_PLAYER_RELATIVE, _UNIT_HP, _UNIT_TYPE, _SELECTED]

def load(path, num_cpu=16):
  """Load act function that was returned by learn function.
  Parameters
  ----------
  path: str
      path to the act function pickle
  num_cpu: int
      number of cpus to use for executing the policy
  Returns
  -------
  act: ActWrapper
      function that takes a batch of observations
      and returns actions.
  """
  return ActWrapper.load(path, num_cpu=num_cpu)

def learn(env,
          q_func,
          num_actions,  # Number of actions (discrete)
          max_num_episodes=150,  # Number of episodes before stopping
          saved_model_path=None,  # Path of the model to save to (and restore)
          continue_model=False,  # Restore a model and continue training it
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          param_noise=False,
          callback=None):

    '''
    See baselines deepq documentation for information.
    Returns ActWrapper to save and load the learned policies
    '''

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        # Shape of observation space
        return U.BatchInput((my_utils.get_screen_channel_size(used_features=_USED_SCREEN_FEATURES), 64, 64), name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=num_actions,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
    )
    act_params = {
        "make_obs_ph": make_obs_ph,
        "q_func": q_func,
        "num_actions": num_actions
    }

    if saved_model_path is not None and continue_model:
        U.load_state(saved_model_path)

    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None

    # Reset environment
    obs = env.reset()
    marine_squad = MarineSquad()

    # Get all relevant observations
    screen_observations = obs[0].observation["screen"][_USED_SCREEN_FEATURES]
    screen_observations = my_utils.preprocess_screen(screen=screen_observations,
                                                     used_features=_USED_SCREEN_FEATURES)

    # print("SHAPE: ", screen_observations.shape)

    reset = True
    model_saved = False
    model_file = saved_model_path

    for t in range(max_timesteps):
        if callback is not None:
            if callback(locals(), globals()):
                break
        # Take action and update exploration to the newest value
        kwargs = {}
        if not param_noise:
            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(
                1. - exploration.value(t) + exploration.value(t) / float(num_actions))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold

        # Select first action in batch
        action = act(np.array(screen_observations)[None],
                     update_eps=update_eps,
                     **kwargs)[0]
        reset = False

        exec_action = marine_squad.select_action(obs[0], action_id=action)

        # Execute action
        obs = env.step(actions=exec_action)

        done = (obs[0].step_type == environment.StepType.LAST)

        if obs[0].observation["player"][_SUPPLY_USED] <= 0:
            # Weird bug where the environment does not stop when all units are dead
            done = environment.StepType.LAST

        rew = obs[0].reward
        new_screen_observations = obs[0].observation["screen"][_USED_SCREEN_FEATURES]
        new_screen_observations = my_utils.preprocess_screen(screen=new_screen_observations,
                                                             used_features=_USED_SCREEN_FEATURES)

        # Save to replay buffer (s,a,r,s')
        replay_buffer.add(screen_observations, action, rew, new_screen_observations, float(done))
        episode_rewards[-1] += rew

        # Update observations!
        screen_observations = new_screen_observations

        if done:
            # Reset
            print("Episode Reward: %s" % episode_rewards[-1])
            obs = env.reset()
            screen_observations = obs[0].observation["screen"][_USED_SCREEN_FEATURES]
            screen_observations = my_utils.preprocess_screen(screen=screen_observations,
                                                             used_features=_USED_SCREEN_FEATURES)
            marine_squad = MarineSquad()
            episode_rewards.append(0.0)
            reset = True

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            if prioritized_replay:
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

        # Update target network periodically.
        if t > learning_starts and t % target_network_update_freq == 0:
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()

        if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if print_freq is not None:
                    logger.log("Saving model.meta due to mean reward increase: {} -> {}".format(
                        saved_mean_reward, mean_100ep_reward))
                U.save_state(model_file)
                model_saved = True
                saved_mean_reward = mean_100ep_reward

        if num_episodes >= max_num_episodes:
            U.save_state(model_file)
            return ActWrapper(act, act_params)

    return ActWrapper(act, act_params)




