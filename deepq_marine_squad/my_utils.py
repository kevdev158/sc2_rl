import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Some of the units in minigames (more will be added throughout the experiment)
# https://pastebin.com/KCwwLiQ1
_T_MARINE = 48
_Z_ZERGLING = 105
_Z_BANELING = 9
_Z_ROACH = 110
_N_MINERAL_SHARD = 597

_USED_UNITS = [_T_MARINE, _Z_BANELING, _Z_ROACH, _Z_ZERGLING, _N_MINERAL_SHARD]

# Code adapted from: https://github.com/XHUJOY/pysc2-agents/blob/master/utils.py
def preprocess_screen(screen, used_features):
    '''
    screen: shape(len(used_features), screen_size, screen_size)
    used_features: indices of the used features (e.g. features.SCREEN_FEATURES.unit_type.index)
    returns: pre-processed features (normalize scalar values, one hot encode categorical values)
    '''
    assert(screen.shape[0] == len(used_features))
    assert(screen.shape[1] == screen.shape[2])
    layers = []
    for i, j in zip(used_features, range(len(used_features))):
        if features.SCREEN_FEATURES[i].index == _SCREEN_UNIT_TYPE:
            layer = np.zeros(shape=(len(_USED_UNITS),
                                    screen.shape[1],
                                    screen.shape[2]),
                             dtype=np.float32)
            for k, l in zip(_USED_UNITS, range(len(_USED_UNITS))):
                ind_y, ind_x = (screen[j] == k).nonzero()
                layer[l, ind_y, ind_x] = 1
            layers.append(layer)
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            # type(screen[j]) != type(screen[j:j+1])
            # Would otherwise return a sz x sz array, want a 1 x sz x sz array
            layers.append(screen[j:j+1] / features.SCREEN_FEATURES[i].scale)
        else:
            # Shape of num_categories x sz x sz
            layer = np.zeros(shape=(features.SCREEN_FEATURES[i].scale,
                                    screen.shape[1],
                                    screen.shape[2]),
                             dtype=np.float32)
            for k in range(features.SCREEN_FEATURES[i].scale):
                ind_y, ind_x = (screen[j] == k).nonzero()
                layer[k, ind_y, ind_x] = 1
            layers.append(layer)

    return np.concatenate(layers, axis=0)


def preprocess_minimap(minimap, used_features):
    '''
    minimap: shape(len(used_features), screen_size, screen_size)
    used_features: indices of the used features (e.g. features.SCREEN_FEATURES.unit_type.index)
    returns: pre-processed features (normalize scalar values, one hot encodes categorical values)
    '''
    assert(minimap.shape[0] == len(used_features))
    assert(minimap.shape[1] == minimap.shape[2])
    layers = []
    for i, j in zip(used_features, range(len(used_features))):
        if features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            layers.append(minimap[j:j+1] / features.MINIMAP_FEATURES[i].scale)
        else:
            layer = np.zeros(shape=(features.MINIMAP_FEATURES[i].scale,
                                    minimap.shape[1],
                                    minimap.shape[2]),
                             dtype=np.float32)
            for k in range(features.MINIMAP_FEATURES[i].scale):
                ind_y, ind_x = (minimap[j] == k).nonzero()
                layer[k, ind_y, ind_x] = 1
            layers.append(layer)

    return np.concatenate(layers, axis=0)


def get_screen_channel_size(used_features):
    c = 0
    for i in used_features:
        if features.SCREEN_FEATURES[i].index == _SCREEN_UNIT_TYPE:
            c += len(_USED_UNITS)
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.SCREEN_FEATURES[i].scale

    return c


def get_minimap_channel_size(used_features):
    c = 0
    for i in used_features:
        if features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            c += 1
        else:
            c += features.MINIMAP_FEATURES[i].scale

    return c

def debug_screen_channel_size():
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
    _USED_SCREEN_FEATURES = [_PLAYER_RELATIVE, _UNIT_HP, _UNIT_TYPE, _SELECTED]
    print(get_screen_channel_size(_USED_SCREEN_FEATURES))

if __name__ == '__main__':
    debug_screen_channel_size()