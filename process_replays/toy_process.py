import sys
import platform
import threading
import multiprocessing
import time

from pysc2.lib import app
from pysc2 import run_configs
from pysc2.bin import replay_actions
from pysc2.lib import gfile

import gflags as flags

FLAGS = flags.FLAGS

'''
Code adapted from: 
https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
'''

#TODO: Get familiar with python's multiprocessing
def main(unused_argv):
    run_config = run_configs.get()

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist".format(FLAGS.replays))

    stats_queue = multiprocessing.Queue()
    stats_thread = threading.Thread(target=replay_actions.stats_printer, args=(stats_queue,))
    stats_thread.start()

    try:
        print("Processing replay: ", FLAGS.replays)
        replay_list = list(run_config.replay_paths(FLAGS.replays))
        replay_queue = multiprocessing.JoinableQueue(10)
        replay_queue_thread = threading.Thread(target=replay_actions.replay_queue_filler,
                                               args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        for i in range(FLAGS.parallel):
            process = replay_actions.ReplayProcessor(proc_id=i,
                                                     run_config=run_config,
                                                     replay_queue=replay_queue,
                                                     stats_queue=stats_queue)
            process.daemon = True
            process.start()
            time.sleep(1)

        replay_queue.join()

    except KeyboardInterrupt:
        print("Exiting")

    finally:
        stats_queue.put(None)  # Tells stats_thread to print and exit
        stats_thread.join()

if __name__ == '__main__':
    app.run(main)


