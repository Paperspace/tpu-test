import os
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

def axy_computation(a, x, y):
  return a * x + y

inputs = [
    3.0,
    tf.ones([3, 3], tf.float32),
    tf.ones([3, 3], tf.float32),
]

tpu_computation = tpu.rewrite(axy_computation, inputs)

tpu_name = os.environ['TPU_NAME']
print('TPU_NAME: %s' % tpu_name)
tpu_grpc_url = os.environ['TPU_GRPC_URL']
print('TPU_GRPC_URL: %s' % tpu_grpc_url)
tpu_model_dir = os.environ['TPU_MODEL_DIR']
print('TPU_MODEL_DIR: %s' % tpu_model_dir)

with tf.Session(tpu_grpc_url) as sess:
  sess.run(tpu.initialize_system())
  sess.run(tf.global_variables_initializer())
  output = sess.run(tpu_computation)
  print(output)
  sess.run(tpu.shutdown_system())

print('Done!')
