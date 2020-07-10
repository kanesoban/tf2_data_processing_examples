from __future__ import absolute_import, division, print_function, unicode_literals

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


root = pathlib.Path(HGG)
t1ce = sorted([str(e) for e in root.glob("*/*t1ce.nii.gz")])
seg = sorted([str(e) for e in root.glob("*/*seg.nii.gz")])


d = tf.data.Dataset.from_tensor_slices(t1ce)

# transform a string tensor to upper case string using a Python function
@tf.function
def decode_str(t: tf.Tensor) -> str:
    return t.numpy().decode('utf-8')


d2 = d.map(lambda x: tf.py_function(func=decode_str, inp=[x], Tout=tf.string))

l = [e for e in d2]
print(l[0].numpy().decode())
