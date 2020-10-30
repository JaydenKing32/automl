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
"""A few predefined label id mapping."""
import tensorflow as tf
import yaml

coco = {
    # 0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}

voc = {
    # 0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

waymo = {
    # 0: 'background',
    1: 'vehicle',
    2: 'pedestrian',
    3: 'cyclist',
}

imagenet = {
    # 0: 'background',
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'axe', 9: 'baby bed',
    10: 'traffic light', 11: 'bagel', 12: 'balance beam', 13: 'crutch', 14: 'band aid', 15: 'bench', 16: 'bird',
    17: 'basketball', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'bee', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'binder', 26: 'baseball', 27: 'backpack', 28: 'bow tie', 29: 'bow', 30: 'croquet ball', 31: 'brassiere',
    32: 'tie', 33: 'armadillo', 34: 'butterfly', 35: 'camel', 36: 'can opener', 37: 'ant', 38: 'cart', 39: 'cattle',
    40: 'cello', 41: 'centipede', 42: 'chain saw', 43: 'dumbbell', 44: 'chime', 45: 'cocktail shaker',
    46: 'coffee maker', 47: 'computer keyboard', 48: 'computer mouse', 49: 'corkscrew', 50: 'cream', 51: 'bowl',
    52: 'banana', 53: 'apple', 54: 'cup or mug', 55: 'orange', 56: 'digital clock', 57: 'dishwasher',
    58: 'bathing cap', 59: 'pizza', 60: 'dragonfly', 61: 'drum', 62: 'chair', 63: 'electric fan', 64: 'bell pepper',
    65: 'face powder', 66: 'fig', 67: 'filing cabinet', 68: 'flower pot', 69: 'flute', 70: 'fox', 71: 'french horn',
    72: 'frog', 73: 'laptop', 74: 'giant panda', 75: 'goldfish', 76: 'golf ball', 77: 'golfcart', 78: 'microwave',
    79: 'guitar', 80: 'toaster', 81: 'hair spray', 82: 'refrigerator', 83: 'hammer', 84: 'hamster', 85: 'harmonica',
    86: 'harp', 87: 'hat with a wide brim', 88: 'head cabbage', 89: 'helmet', 90: 'hippopotamus', 91: 'horizontal bar',
    92: 'beaker', 93: 'hotdog', 94: 'iPod', 95: 'isopod', 96: 'jellyfish', 97: 'koala bear', 98: 'ladle',
    99: 'ladybug', 100: 'lamp', 101: 'frying pan', 102: 'lemon', 103: 'lion', 104: 'lipstick', 105: 'lizard',
    106: 'lobster', 107: 'maillot', 108: 'maraca', 109: 'microphone', 110: 'guacamole', 111: 'milk can',
    112: 'miniskirt', 113: 'monkey', 114: 'antelope', 115: 'mushroom', 116: 'nail', 117: 'neck brace', 118: 'oboe',
    119: 'diaper', 120: 'otter', 121: 'pencil box', 122: 'pencil sharpener', 123: 'perfume', 124: 'accordion',
    125: 'piano', 126: 'pineapple', 127: 'ping-pong ball', 128: 'pitcher', 129: 'domestic cat', 130: 'plastic bag',
    131: 'plate rack', 132: 'pomegranate', 133: 'popsicle', 134: 'porcupine', 135: 'power drill', 136: 'pretzel',
    137: 'printer', 138: 'puck', 139: 'punching bag', 140: 'purse', 141: 'rabbit', 142: 'racket', 143: 'ray',
    144: 'red panda', 145: 'hamburger', 146: 'remote control', 147: 'rubber eraser', 148: 'rugby ball', 149: 'ruler',
    150: 'salt or pepper shaker', 151: 'saxophone', 152: 'scorpion', 153: 'screwdriver', 154: 'seal', 155: 'banjo',
    156: 'ski', 157: 'skunk', 158: 'snail', 159: 'snake', 160: 'snowmobile', 161: 'snowplow', 162: 'soap dispenser',
    163: 'soccer ball', 164: 'sofa', 165: 'spatula', 166: 'squirrel', 167: 'starfish', 168: 'stethoscope',
    169: 'stove', 170: 'strainer', 171: 'strawberry', 172: 'stretcher', 173: 'sunglasses', 174: 'swimming trunks',
    175: 'swine', 176: 'syringe', 177: 'table', 178: 'tape player', 179: 'tennis ball', 180: 'tick', 181: 'burrito',
    182: 'tiger', 183: 'hair dryer', 184: 'bookshelf', 185: 'artichoke', 186: 'trombone', 187: 'trumpet',
    188: 'turtle', 189: 'tv or monitor', 190: 'unicycle', 191: 'vacuum', 192: 'violin', 193: 'volleyball',
    194: 'waffle iron', 195: 'washer', 196: 'water bottle', 197: 'watercraft', 198: 'whale', 199: 'wine bottle',
    200: 'cucumber'
}


def get_label_map(mapping):
    """Get label id map based on the name, filename, or dict."""
    # case 1: if it is None or dict, just return it.
    if not mapping or isinstance(mapping, dict):
        return mapping

    # case 2: if it is a yaml file, load it to a dict and return the dict.
    assert isinstance(mapping, str), 'mapping must be dict or str.'
    if mapping.endswith('.yaml'):
        with tf.io.gfile.GFile(mapping) as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    # case 3: it is a name of a predefined dataset.
    return {'coco': coco, 'voc': voc, 'waymo': waymo, 'imagenet': imagenet}[mapping]
