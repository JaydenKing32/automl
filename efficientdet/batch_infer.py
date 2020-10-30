# https://github.com/google/automl/issues/82#issuecomment-605539515

from PIL import Image
import os
import inference
import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

MODEL = 'efficientdet-d0'
MODEL_DIR = 'efficientdet-d0'
driver = inference.ServingDriver(MODEL, MODEL_DIR)
driver.build()
i = 0
for f in tf.io.gfile.glob('./data/ILSVRC2013/*.JPEG'):
    if i > 100:
        break
    image = np.array(Image.open(f))
    if len(image.shape) < 3:
        print(f)
        continue
    predictions = driver.serve_images([image])
    out_image = driver.visualize(
        image,
        predictions[0],
        min_score_thresh=0.5,
        max_boxes_to_draw=10,
    )
    i = i + 1
    output_image_path = os.path.join('./out', f"{i:05d}.jpg")
    print(output_image_path)
    Image.fromarray(out_image).save(output_image_path)
