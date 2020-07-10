from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

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

NUM_PARALLEL_READS = 4
SIZE = 240
N_SLICES = 155
MAX_T1CE = 8508
N_CLASSES = 4

# Slice can be from 0 to 154
SLICE = 80

root = pathlib.Path(HGG)

t1ce_ds = tf.data.Dataset.list_files(str(root / '*/*t1ce.nii.gz'), shuffle=False)
seg_ds = tf.data.Dataset.list_files(str(root / '*/*seg.nii.gz'), shuffle=False)


@tf.function
def preprocess_data(path):
    return nib.load(path.numpy().decode()).get_data()[:, :, SLICE].reshape((SIZE, SIZE, 1)) / MAX_T1CE


@tf.function
def preprocess_data2(path):
    return tf.data.Dataset.from_tensor_slices(nib.load(path.numpy().decode()).get_data()[:, :, :].reshape((-1, SIZE, SIZE, 1)) / MAX_T1CE)


@tf.function
def preprocess_label(path):
    return nib.load(path.numpy().decode()).get_data()[:, :, SLICE].reshape((SIZE, SIZE)).astype(np.int)

@tf.function
def preprocess_label2(path):
    return tf.data.Dataset.from_tensor_slices(nib.load(path.numpy().decode()).get_data()[:, :, :].reshape((-1, SIZE, SIZE)).astype(np.int))


# Compare speed with and without prefetch

# tensorflow.python.framework.errors_impl.DataLossError: corrupted record at 0 [Op:IteratorGetNextSync]
#t1ce_ds = t1ce_ds.interleave(tf.data.TFRecordDataset, cycle_length=NUM_PARALLEL_READS, num_parallel_calls=tf.data.experimental.AUTOTUNE)


t1ce_ds = t1ce_ds.flat_map(preprocess_data2)

'''
t1ce_ds = t1ce_ds.map(lambda x: tf.py_function(func=preprocess_data, inp=[x], Tout=tf.float32),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
'''

t1ce_ds = t1ce_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# tensorflow.python.framework.errors_impl.DataLossError: corrupted record at 0 [Op:IteratorGetNextSync]
#seg_ds = seg_ds.interleave(tf.data.TFRecordDataset, cycle_length=NUM_PARALLEL_READS, num_parallel_calls=tf.data.experimental.AUTOTUNE)

seg_ds = seg_ds.flat_map(preprocess_label2)

'''
seg_ds = seg_ds.map(lambda x: tf.py_function(func=preprocess_label, inp=[x], Tout=tf.int32),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
'''

seg_ds = seg_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

seg_ds = seg_ds.map(lambda x: tf.one_hot(x, N_CLASSES), num_parallel_calls=tf.data.experimental.AUTOTUNE)

labeled_ds = tf.data.Dataset.zip((t1ce_ds, seg_ds))

dataset = labeled_ds.shuffle(buffer_size=100).batch(5)

max_val = 0
for data, label in dataset:
    max_val = max(max_val, np.max(data))

print(max_val)
