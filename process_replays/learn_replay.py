import os

import gflags as flags

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller

from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS

flags.DEFINE_string("replays", None, "Path to a directory of replays.")
flags.DEFINE_integer("step_mul", 8, "How many game steps per observation.")

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

class ReplayPipeline:
    def __init__(self, replay_path, run_config, replay_cache):
        self.replay_path = replay_path
        self.run_config = run_config
        self.replay_cache = replay_cache
        self.replay_meta = {}  # Try to log races, map, results

    @staticmethod
    def valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 1000 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            return False
        for p in info.player_info:
            if p.player_apm < 10 or p.player_mmr < 1000:
                # Low APM = player just standing around.
                # Low MMR = corrupt replay or player who is weak.
                return False
        return True

    def setup_replay(self):
        #TODO: need a function to expose parts to model
        try:
            with self.run_config.start() as controller:
                ping = controller.ping()
                for _ in range(300):
                    try:
                        print("Setting up replay...")
                        replay_name = os.path.basename(self.replay_path)[:10]
                        replay_data = self.run_config.replay_data(self.replay_path)
                        info = controller.replay_info(replay_data)

                        print("Trying to process ", replay_name)
                        if self.valid_replay(info, ping):
                            # Do initial logging here
                            map_data = None
                            if info.local_map_path:
                                map_data = self.run_config.map_data(info.local_map_path)
                            for player_id in [1, 2]:
                                self.process_replay(controller, replay_data,
                                                    map_data, player_id)
                        else:
                            print("Replay is invalid")
        except (protocol.ConnectionError, protocol.ProtocolError,
              remote_controller.RequestError):
            print("Replay crashed.")

        except KeyboardInterrupt:
            return



    def process_replay(self, controller, replay_data, map_data, player_id):
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id
        ))

        feat = features.Features(controller.game_info())
        controller.step()

        # Get rid of this while loop -- too sleepy
        while True:
            obs = controller.observe




def main(unused_argv):
    size = point.Point(64, 64)
    interface = sc_pb.InterfaceOptions(
        raw=True, score=False,
        feature_layer=sc_pb.SpatialCameraSetup(width=24))
    size.assign_to(interface.feature_layer.resolution)
    size.assign_to(interface.feature_layer.minimap_resolution)

    run_config = run_configs.get()
    replay_list = [run_config.replay_paths(FLAGS.replays)]

    # maximum number of steps per replay -- sample first ten minutes
    max_episode_steps = 9600  # step size is 8
    batch_size = 40  # last five seconds
    replay_cache = []  # Should be an array of size -1 * 64 * 64 * features used
    for replay_path in replay_list:
        replay_pipeline = ReplayPipeline(replay_path=replay_path,
                                         run_config=run_config,
                                         replay_cache=replay_cache)
        replay_pipeline.process_replay()



