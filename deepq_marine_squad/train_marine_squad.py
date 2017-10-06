import os
import sys

import gflags as flags
import deepq_marine_squad

from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions

FLAGS = flags.FLAGS
step_mul = 4

def main():
    FLAGS(sys.argv)

    with sc2_env.SC2Env(
        "DefeatZerglingsAndBanelings",
        step_mul=step_mul,
        visualize=True
    ) as env:
        model = deepq.models.cnn_to_mlp(
            convs=[
                (64, 8, 4),
                (64, 4, 2),
                (32, 8, 4),
                (32, 4, 2),
            ],
            hiddens=[256],
            dueling=True
        )

        act = deepq_marine_squad.learn(
            env=env,
            q_func=model,
            num_actions=15,
            max_num_episodes=180,  # Training on episodes rather than total steps
            saved_model_path=os.path.join("saved_models", "zerglings_and_banelings"),
            continue_model=False,  # Load exisiting model from saved_model path and continue training it
            lr=2.5e-4,
            max_timesteps=1000000,
            buffer_size=50000,
            exploration_fraction=0.25,
            exploration_final_eps=0.01,
            train_freq=5,
            learning_starts=200,
            target_network_update_freq=500,
            gamma=0.99,
            prioritized_replay=True,
            checkpoint_freq=10000   # Total steps before updating model
        )

        act.save("zerglings_and_banelings.pkl")

if __name__ == "__main__":
    main()