# picks up the GPU it seems
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#print(os.environ.get("CUDA_PATH"))

#os.environ["CUDA_PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"


#print(os.environ.get("CUDA_PATH"))


print(device_lib.list_local_devices())


#tf.test_is_gpu_available() 