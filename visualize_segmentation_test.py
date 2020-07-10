from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Is this eager mode ?
tf.executing_eagerly()
assert (tf.__version__ == '2.0.0')
tf.config.experimental_run_functions_eagerly(True)

from os.path import join
import pathlib
import nibabel as nib


DATASET_ROOT = '/home/cszsolnai/datasets/MICCAI_BraTS_2018_Data_Training'
HGG = join(DATASET_ROOT, 'HGG')
LGG = join(DATASET_ROOT, 'LGG')

N_CLASSES = 4
MAX_T1CE = 8508
SIZE = 240
# Slice can be from 0 to 154
SLICE = 80

root = pathlib.Path(HGG)

t1ce_ds = tf.data.Dataset.list_files(str(root / '*/*t1ce.nii.gz'), shuffle=False)
#seg_ds = tf.data.Dataset.list_files(str(root / '*/*seg.nii.gz'), shuffle=False)


@tf.function
def preprocess_data(path):
    return nib.load(path.numpy().decode()).get_data()[:, :, SLICE].reshape((SIZE, SIZE, 1)) / MAX_T1CE

t1ce_ds = t1ce_ds.map(lambda x: tf.py_function(func=preprocess_data, inp=[x], Tout=tf.float32),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

model = tf.keras.models.load_model(
    'models/model.h5',
    custom_objects=None,
    compile=False
)

data = None
for image in t1ce_ds:
    data = image
    break

predicted = model.predict(tf.expand_dims(data, axis=0))[0]
images = [data, tf.expand_dims(predicted.max(axis=2), axis=2)]

titles = ['Input image', 'Mask']
for i, title in enumerate(titles):
    plt.subplot(1, 2, i+1)
    plt.title(titles[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i]))
    plt.axis('off')
plt.show()
