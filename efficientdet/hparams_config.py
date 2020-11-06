# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hparams for model architecture and trainer."""
import ast
import collections
import copy
from typing import Any, Dict, Text
import six
import tensorflow as tf
import yaml


def eval_str_fn(val):
  if val in {'true', 'false'}:
    return val == 'true'
  try:
    return ast.literal_eval(val)
  except (ValueError, SyntaxError):
    return val


# pylint: disable=protected-access
class Config(object):
  """A config utility class."""

  def __init__(self, config_dict=None):
    self.update(config_dict)

  def __setattr__(self, k, v):
    self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)

  def __getattr__(self, k):
    return self.__dict__[k]

  def __getitem__(self, k):
    return self.__dict__[k]

  def __repr__(self):
    return repr(self.as_dict())

  def __str__(self):
    try:
      return yaml.dump(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True):
    """Recursively update internal members."""
    if not config_dict:
      return

    for k, v in six.iteritems(config_dict):
      if k not in self.__dict__:
        if allow_new_keys:
          self.__setattr__(k, v)
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
          self.__dict__[k]._update(v.as_dict(), allow_new_keys)
        else:
          self.__setattr__(k, v)

  def get(self, k, default_value=None):
    return self.__dict__.get(k, default_value)

  def update(self, config_dict):
    """Update members while allowing new keys."""
    self._update(config_dict, allow_new_keys=True)

  def keys(self):
    return self.__dict__.keys()

  def override(self, config_dict_or_str, allow_new_keys=False):
    """Update members while disallowing new keys."""
    if isinstance(config_dict_or_str, str):
      if not config_dict_or_str:
        return
      elif '=' in config_dict_or_str:
        config_dict = self.parse_from_str(config_dict_or_str)
      elif config_dict_or_str.endswith('.yaml'):
        config_dict = self.parse_from_yaml(config_dict_or_str)
      else:
        raise ValueError(
            'Invalid string {}, must end with .yaml or contains "=".'.format(
                config_dict_or_str))
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys)

  def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
      return config_dict

  def save_to_yaml(self, yaml_file_path):
    """Write a dictionary into a yaml file."""
    with tf.io.gfile.GFile(yaml_file_path, 'w') as f:
      yaml.dump(self.as_dict(), f, default_flow_style=False)

  def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
    """Parse a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        if not kv_pair:  # skip empty string
          continue
        key_str, value_str = kv_pair.split('=')
        key_str = key_str.strip()

        def add_kv_recursive(k, v):
          """Recursively parse x.y.z=tt to {x: {y: {z: tt}}}."""
          if '.' not in k:
            if '*' in v:
              # we reserve * to split arrays.
              return {k: [eval_str_fn(vv) for vv in v.split('*')]}
            return {k: eval_str_fn(v)}
          pos = k.index('.')
          return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}

        def merge_dict_recursive(target, src):
          """Recursively merge two nested dictionary."""
          for k in src.keys():
            if ((k in target and isinstance(target[k], dict) and
                 isinstance(src[k], collections.abc.Mapping))):
              merge_dict_recursive(target[k], src[k])
            else:
              target[k] = src[k]

        merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
      return config_dict
    except ValueError:
      raise ValueError('Invalid config_str: {}'.format(config_str))

  def as_dict(self):
    """Returns a dict representation."""
    config_dict = {}
    for k, v in six.iteritems(self.__dict__):
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict
    # pylint: enable=protected-access


def default_detection_configs():
  """Returns a default detection configs."""
  h = Config()

  # model name.
  h.name = 'efficientdet-d1'

  # activation type: see activation_fn in utils.py.
  h.act_type = 'swish'

  # input preprocessing parameters
  h.image_size = 640  # An integer or a string WxH such as 640x320.
  h.target_size = None
  h.input_rand_hflip = True
  h.jitter_min = 0.1
  h.jitter_max = 2.0
  h.autoaugment_policy = None
  h.use_augmix = False
  h.grid_mask = False
  # mixture_width, mixture_depth, alpha
  h.augmix_params = [3, -1, 1]
  h.sample_image = None

  # dataset specific parameters
  # TODO(tanmingxing): update this to be 91 for COCO, and 21 for pascal.
  h.num_classes = 90  # 1+ actual classes, 0 is reserved for background.
  h.seg_num_classes = 3  # segmentation classes
  h.heads = ['object_detection']  # 'object_detection', 'segmentation'

  h.skip_crowd_during_training = True
  # h.label_map = None  # a dict or a string of 'coco', 'voc', 'waymo'.
  # h.label_map = 'coco'
  # h.label_map = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 6: 'bus', 7: 'train', 8: 'tvmonitor', 9: 'boat', 11: 'diningtable', 12: 'sofa', 13: 'pottedplant', 14: 'motorbike', 15: 'aeroplane', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 44: 'bottle', 62: 'chair'}
  # h.label_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'axe', 9: 'baby bed', 10: 'traffic light', 11: 'bagel', 12: 'balance beam', 13: 'crutch', 14: 'band aid', 15: 'bench', 16: 'bird', 17: 'basketball', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'bee', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'binder', 26: 'baseball', 27: 'backpack', 28: 'bow tie', 29: 'bow', 30: 'croquet ball', 31: 'brassiere', 32: 'tie', 33: 'armadillo', 34: 'butterfly', 35: 'camel', 36: 'can opener', 37: 'ant', 38: 'cart', 39: 'cattle', 40: 'cello', 41: 'centipede', 42: 'chain saw', 43: 'dumbbell', 44: 'chime', 45: 'cocktail shaker', 46: 'coffee maker', 47: 'computer keyboard', 48: 'computer mouse', 49: 'corkscrew', 50: 'cream', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'cup or mug', 55: 'orange', 56: 'digital clock', 57: 'dishwasher', 58: 'bathing cap', 59: 'pizza', 60: 'dragonfly', 61: 'drum', 62: 'chair', 63: 'electric fan', 64: 'bell pepper', 65: 'face powder', 66: 'fig', 67: 'filing cabinet', 68: 'flower pot', 69: 'flute', 70: 'fox', 71: 'french horn', 72: 'frog', 73: 'laptop', 74: 'giant panda', 75: 'goldfish', 76: 'golf ball', 77: 'golfcart', 78: 'microwave', 79: 'guitar', 80: 'toaster', 81: 'hair spray', 82: 'refrigerator', 83: 'hammer', 84: 'hamster', 85: 'harmonica', 86: 'harp', 87: 'hat with a wide brim', 88: 'head cabbage', 89: 'helmet', 90: 'hippopotamus', 91: 'horizontal bar', 92: 'beaker', 93: 'hotdog', 94: 'iPod', 95: 'isopod', 96: 'jellyfish', 97: 'koala bear', 98: 'ladle', 99: 'ladybug', 100: 'lamp', 101: 'frying pan', 102: 'lemon', 103: 'lion', 104: 'lipstick', 105: 'lizard', 106: 'lobster', 107: 'maillot', 108: 'maraca', 109: 'microphone', 110: 'guacamole', 111: 'milk can', 112: 'miniskirt', 113: 'monkey', 114: 'antelope', 115: 'mushroom', 116: 'nail', 117: 'neck brace', 118: 'oboe', 119: 'diaper', 120: 'otter', 121: 'pencil box', 122: 'pencil sharpener', 123: 'perfume', 124: 'accordion', 125: 'piano', 126: 'pineapple', 127: 'ping-pong ball', 128: 'pitcher', 129: 'domestic cat', 130: 'plastic bag', 131: 'plate rack', 132: 'pomegranate', 133: 'popsicle', 134: 'porcupine', 135: 'power drill', 136: 'pretzel', 137: 'printer', 138: 'puck', 139: 'punching bag', 140: 'purse', 141: 'rabbit', 142: 'racket', 143: 'ray', 144: 'red panda', 145: 'hamburger', 146: 'remote control', 147: 'rubber eraser', 148: 'rugby ball', 149: 'ruler', 150: 'salt or pepper shaker', 151: 'saxophone', 152: 'scorpion', 153: 'screwdriver', 154: 'seal', 155: 'banjo', 156: 'ski', 157: 'skunk', 158: 'snail', 159: 'snake', 160: 'snowmobile', 161: 'snowplow', 162: 'soap dispenser', 163: 'soccer ball', 164: 'sofa', 165: 'spatula', 166: 'squirrel', 167: 'starfish', 168: 'stethoscope', 169: 'stove', 170: 'strainer', 171: 'strawberry', 172: 'stretcher', 173: 'sunglasses', 174: 'swimming trunks', 175: 'swine', 176: 'syringe', 177: 'table', 178: 'tape player', 179: 'tennis ball', 180: 'tick', 181: 'burrito', 182: 'tiger', 183: 'hair dryer', 184: 'bookshelf', 185: 'artichoke', 186: 'trombone', 187: 'trumpet', 188: 'turtle', 189: 'tv or monitor', 190: 'unicycle', 191: 'vacuum', 192: 'violin', 193: 'volleyball', 194: 'waffle iron', 195: 'washer', 196: 'water bottle', 197: 'watercraft', 198: 'whale', 199: 'wine bottle', 200: 'cucumber'}
  # h.label_map = {0: 'background', 1: 'sign'}
  h.max_instances_per_image = 100  # Default to 100 for COCO.
  h.regenerate_source_id = False

  # model architecture
  h.min_level = 3
  h.max_level = 7
  h.num_scales = 3
  # ratio w/h: 2.0 means w=1.4, h=0.7. Can be computed with k-mean per dataset.
  h.aspect_ratios = [1.0, 2.0, 0.5]
  h.anchor_scale = 4.0
  # is batchnorm training mode
  h.is_training_bn = True
  # optimization
  h.momentum = 0.9
  h.optimizer = 'sgd'  # can be 'adam' or 'sgd'.
  h.learning_rate = 0.08  # 0.008 for adam.
  h.lr_warmup_init = 0.008  # 0.0008 for adam.
  h.lr_warmup_epoch = 1.0
  h.first_lr_drop_epoch = 200.0
  h.second_lr_drop_epoch = 250.0
  h.poly_lr_power = 0.9
  h.clip_gradients_norm = 10.0
  h.num_epochs = 300
  h.data_format = 'channels_last'

  # classification loss
  h.label_smoothing = 0.0  # 0.1 is a good default
  # Behold the focal loss parameters
  h.alpha = 0.25
  h.gamma = 1.5

  # localization loss
  h.delta = 0.1  # regularization parameter of huber loss.
  # total loss = box_loss * box_loss_weight + iou_loss * iou_loss_weight
  h.box_loss_weight = 50.0
  h.iou_loss_type = None
  h.iou_loss_weight = 1.0

  # regularization l2 loss.
  h.weight_decay = 4e-5
  h.strategy = None  # 'tpu', 'gpus', None
  h.mixed_precision = False  # If False, use float32.
  h.model_optimizations = {}  # 'prune'

  # For detection.
  h.box_class_repeats = 3
  h.fpn_cell_repeats = 3
  h.fpn_num_filters = 88
  h.separable_conv = True
  h.apply_bn_for_resampling = True
  h.conv_after_downsample = False
  h.conv_bn_act_pattern = False
  h.drop_remainder = True  # drop remainder for the final batch eval.

  # For post-processing nms, must be a dict.
  h.nms_configs = {
      'method': 'gaussian',
      'iou_thresh': None,  # use the default value based on method.
      'score_thresh': None,
      'sigma': None,
      'max_nms_inputs': 0,
      'max_output_size': 100,
  }

  # version.
  h.fpn_name = None
  h.fpn_weight_method = None
  h.fpn_config = None

  # No stochastic depth in default.
  h.survival_prob = None
  h.img_summary_steps = None

  h.lr_decay_method = 'cosine'
  h.moving_average_decay = 0.9998
  h.ckpt_var_scope = None  # ckpt variable scope.
  # If true, skip loading pretrained weights if shape mismatches.
  h.skip_mismatch = True

  h.backbone_name = 'efficientnet-b1'
  h.backbone_config = None
  h.var_freeze_expr = None

  # A temporary flag to switch between legacy and keras models.
  h.use_keras_model = True
  h.dataset_type = None
  h.positives_momentum = None

  # For device specific options.
  h.device = {
      # If true, apply gradient checkpointing to reduce memory usage.
      'grad_ckpting': False,
      # All ops in the list will be checkpointed, such as Add/Mul/Conv2d/Floor/
      # Sigmoid and other ops, or more specific, e.g. blocks_10/se/conv2d_1.
      # Adding more ops will save more memory at the cost of more computation.
      # For EfficientDet, [Add_, AddN] reduces mbconv memory to one third
      # with one third more compute, in particular enabling training d6 with
      # batch size 2 on 11Gb (2080ti).
      'grad_ckpting_list': ['Add_', 'AddN'],
      # enable memory logging for NVIDIA cards.
      'nvgpu_logging': False,
  }

  return h


efficientdet_model_param_dict = {
    'efficientdet-d0':
        dict(
            name='efficientdet-d0',
            backbone_name='efficientnet-b0',
            image_size=512,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
        ),
    'efficientdet-d1':
        dict(
            name='efficientdet-d1',
            backbone_name='efficientnet-b1',
            image_size=640,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
        ),
    'efficientdet-d2':
        dict(
            name='efficientdet-d2',
            backbone_name='efficientnet-b2',
            image_size=768,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
        ),
    'efficientdet-d3':
        dict(
            name='efficientdet-d3',
            backbone_name='efficientnet-b3',
            image_size=896,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
        ),
    'efficientdet-d4':
        dict(
            name='efficientdet-d4',
            backbone_name='efficientnet-b4',
            image_size=1024,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d5':
        dict(
            name='efficientdet-d5',
            backbone_name='efficientnet-b5',
            image_size=1280,
            fpn_num_filters=288,
            fpn_cell_repeats=7,
            box_class_repeats=4,
        ),
    'efficientdet-d6':
        dict(
            name='efficientdet-d6',
            backbone_name='efficientnet-b6',
            image_size=1280,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
    'efficientdet-d7':
        dict(
            name='efficientdet-d7',
            backbone_name='efficientnet-b6',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=5.0,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
    'efficientdet-d7x':
        dict(
            name='efficientdet-d7x',
            backbone_name='efficientnet-b7',
            image_size=1536,
            fpn_num_filters=384,
            fpn_cell_repeats=8,
            box_class_repeats=5,
            anchor_scale=4.0,
            max_level=8,
            fpn_weight_method='sum',  # Use unweighted sum for stability.
        ),
}

efficientdet_lite_param_dict = {
    # lite models are in progress and subject to changes.
    'efficientdet-lite0':
        dict(
            name='efficientdet-lite0',
            backbone_name='efficientnet-lite0',
            image_size=320,
            fpn_num_filters=64,
            fpn_cell_repeats=3,
            box_class_repeats=3,
            act_type='relu',
            fpn_weight_method='sum',
            anchor_scale=3.0,
        ),
    'efficientdet-lite1':
        dict(
            name='efficientdet-lite1',
            backbone_name='efficientnet-lite1',
            image_size=384,
            fpn_num_filters=88,
            fpn_cell_repeats=4,
            box_class_repeats=3,
            act_type='relu',
            fpn_weight_method='sum',
            anchor_scale=3.0,
        ),
    'efficientdet-lite2':
        dict(
            name='efficientdet-lite2',
            backbone_name='efficientnet-lite2',
            image_size=448,
            fpn_num_filters=112,
            fpn_cell_repeats=5,
            box_class_repeats=3,
            act_type='relu',
            fpn_weight_method='sum',
            anchor_scale=3.0,
        ),
    'efficientdet-lite3':
        dict(
            name='efficientdet-lite3',
            backbone_name='efficientnet-lite3',
            image_size=512,
            fpn_num_filters=160,
            fpn_cell_repeats=6,
            box_class_repeats=4,
            act_type='relu',
            fpn_weight_method='sum',
        ),
    'efficientdet-lite4':
        dict(
            name='efficientdet-lite4',
            backbone_name='efficientnet-lite4',
            image_size=512,
            fpn_num_filters=224,
            fpn_cell_repeats=7,
            box_class_repeats=4,
            act_type='relu',
            fpn_weight_method='sum',
        ),
}


def get_efficientdet_config(model_name='efficientdet-d1'):
  """Get the default config for EfficientDet based on model name."""
  h = default_detection_configs()
  if model_name in efficientdet_model_param_dict:
    h.override(efficientdet_model_param_dict[model_name])
  elif model_name in efficientdet_lite_param_dict:
    h.override(efficientdet_lite_param_dict[model_name])
  else:
    raise ValueError('Unknown model name: {}'.format(model_name))

  return h


def get_detection_config(model_name):
  if model_name.startswith('efficientdet'):
    return get_efficientdet_config(model_name)
  else:
    raise ValueError('model name must start with efficientdet.')
