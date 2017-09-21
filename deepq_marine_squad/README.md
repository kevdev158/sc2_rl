### Preprocessing
The code for preprocessing features is adapted from: https://github.com/XHUJOY/pysc2-agents/blob/master/utils.py.

The preprocessing will only take into account of the features we expect to see from minigames, since this experiment
is mainly for minigames.
The one hot encoding matrices from categorical features such as `unit_type` will be very sparse.
This will also run reasonably well on computers without GPUs.
