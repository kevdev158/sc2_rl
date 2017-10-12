import os
import sys
import signal
import threading
import multiprocessing
import time
import queue

from pysc2.lib import app
from pysc2 import run_configs
from pysc2.bin import replay_actions
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import gfile
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from s2clientprotocol import sc2api_pb2 as sc_pb
from pprint import pprint

import gflags as flags

FLAGS = flags.FLAGS

size = point.Point(64, 64)
interface = sc_pb.InterfaceOptions(
    raw=True, score=False,
    feature_layer=sc_pb.SpatialCameraSetup(width=24))
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)

'''
Code adapted from: 
https://github.com/deepmind/pysc2/blob/master/pysc2/bin/replay_actions.py
'''

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

_USED_SCREEN_FEATURES = [_PLAYER_RELATIVE, _UNIT_HP, _UNIT_TYPE,
                         _SELECTED, _VISIBILITY]

_USED_MINIMAP_FEATURES = [_MMAP_VISIBILITY, _MMAP_CAMERA,
                          _MMAP_PLAYER_RELATIVE, _MMAP_PLAYER_ID, _MMAP_SELECTED]


class MyReplayProcessor(replay_actions.ReplayProcessor):
    def _update_stage(self, stage):
        self.stats.update(stage)

    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())
        self._update_stage("spawn")
        replay_name = "none"
        while True:
            self._print("Starting up a new SC2 instance")
            self._update_stage("launch")
            try:
                with self.run_config.start() as controller:
                    self._print("SC2 Started successfully")
                    ping = controller.ping()
                    for _ in range(300):

                        # See if there are anymore replays to process
                        try:
                            replay_path = self.replay_queue.get()
                        except queue.Empty:
                            self._update_stage("done")
                            self._print("Empty queue, returning")
                            return

                        try:
                            replay_name = os.path.basename(replay_path)[:10]
                            self.stats.replay = replay_name

                            self._update_stage("open_replay_file")
                            replay_data = self.run_config.replay_data(replay_path)

                            self._update_stage("replay_info")
                            info = controller.replay_info(replay_data)

                            if replay_actions.valid_replay(info, ping):
                                self.stats.replay_stats.maps[info.map_name] += 1
                                for player_info in info.player_info:
                                    self.stats.replay_stats.races[
                                        sc_pb.Race.Name(player_info.player_info.race_actual)] += 1

                                map_data = None

                                if info.local_map_path:
                                    self._update_stage("open map file")
                                    map_data = self.run_config.map_data(info.local_map_path)

                                # Here is where we actually process the replay
                                for player_id in [1, 2]:
                                    self.process_replay(controller, replay_data,
                                                        map_data, player_id)
                            else:
                                self._print("Replay is invalid")
                                self.stats.replay_stats.invalid_replays.add(replay_name)

                        finally:
                            self.replay_queue.task_done()

                    self._update_stage("shutdown")

            except (protocol.ConnectionError, protocol.ProtocolError,
                    remote_controller.RequestError):
                self.stats.replay_stats.crashing_replays.add(replay_name)
            except KeyboardInterrupt:
                return

    def process_replay(self, controller, replay_data, map_data, player_id):
        self._update_stage("start_replay")
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id
        ))

        feat = features.Features(controller.game_info())

        self.stats.replay_stats.replays += 1
        self._update_stage("step")
        controller.step()

        while True:
            self.stats.replay_stats.steps += 1
            self._update_stage("observe")

            obs = controller.observe()

            feature_obs = feat.transform_obs(obs.observation)
            print(feature_obs["screen"][_USED_SCREEN_FEATURES].shape)
            print(feature_obs["minimap"][_USED_MINIMAP_FEATURES].shape)
            exit()

            for action in obs.actions:
                act_fl = action.action_feature_layer
                if act_fl.HasField("unit_command"):
                    self.stats.replay_stats.made_abilities[
                        act_fl.unit_command.ability_id] += 1
                if act_fl.HasField("camera_move"):
                    self.stats.replay_stats.camera_move += 1
                if act_fl.HasField("unit_selection_point"):
                    self.stats.replay_stats.select_pt += 1
                if act_fl.HasField("unit_selection_rect"):
                    self.stats.replay_stats.select_rect += 1
                if action.action_ui.HasField("control_group"):
                    self.stats.replay_stats.control_group += 1

                try:
                    func = feat.reverse_action(action).function
                except ValueError:
                    func = -1

                self.stats.replay_stats.made_actions[func] += 1

            for valid in obs.observation.abilities:
                self.stats.replay_stats.valid_abilities[valid.ability_id] += 1

            for u in obs.observation.raw_data.units:
                self.stats.replay_stats.unit_ids[u.unit_type] += 1

            for ability_id in feat.available_actions(obs.observation):
                self.stats.replay_stats.valid_actions[ability_id] += 1

            if obs.player_result:
                break

            self._update_stage("step")
            controller.step(FLAGS.step_mul)


#TODO: Get familiar with python's multiprocessing
def main(unused_argv):
    run_config = run_configs.get()

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist".format(FLAGS.replays))

    # stats_queue = multiprocessing.Queue()
    # stats_thread = threading.Thread(target=replay_actions.stats_printer, args=(stats_queue,))
    # stats_thread.start()

    try:
        print("Processing replay: ", FLAGS.replays)
        replay_list = list(run_config.replay_paths(FLAGS.replays))
        replay_queue = multiprocessing.JoinableQueue(10)
        replay_queue_thread = threading.Thread(target=replay_actions.replay_queue_filler,
                                               args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        for i in range(FLAGS.parallel):
            process = MyReplayProcessor(proc_id=i,
                                        run_config=run_config,
                                        replay_queue=replay_queue,
                                        stats_queue=[])
            process.daemon = True
            process.start()
            time.sleep(1)

        replay_queue.join()

    except KeyboardInterrupt:
        print("Exiting")

    # finally:
        # stats_queue.put(None)  # Tells stats_thread to print and exit
        # stats_thread.join()

if __name__ == '__main__':
    app.run(main)


