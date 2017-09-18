import os
import sys

import gflags as flags
import deepq_marine_squad

from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

FLAGS = flags.FLAGS
step_mul = 8

def main():
    FLAGS(sys.argv)

    with sc2_env.SC2Env(
        "DefeatZerglingsAndBanelings",
        step_mul=step_mul,
        visualize=True,
        game_steps_per_episode=6000 * step_mul
    ) as env:
        model = deepq.models.cnn_to_mlp(
            convs=[
                (64, 8, 4),
                (64, 4, 2),
                (32, 8, 4),
                (32, 4, 2),
                (16, 8, 4)
            ],
            hiddens=[256, 128, 64],
            dueling=True
        )

        act = deepq_marine_squad.learn(
            env=env,
            q_func=model,
            num_actions=14,
            max_num_episodes=500,
            saved_model_path=os.path.join("saved_models", "my_model"),
            continue_model=True,
            lr=2.5e-4,
            max_timesteps=1000000,
            buffer_size=50000,
            exploration_fraction=0.15,
            exploration_final_eps=0.01,
            train_freq=5,
            learning_starts=200,
            target_network_update_freq=500,
            gamma=0.99,
            prioritized_replay=True,
            checkpoint_freq=10000
        )

        act.save("marine_roaches.pkl")

if __name__ == "__main__":
    main()