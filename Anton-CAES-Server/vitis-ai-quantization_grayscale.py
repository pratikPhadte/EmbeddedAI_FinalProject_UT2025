import tensorflow as tf
import numpy as np
import importlib
import common
import cv2 
import time
import os
import pickle

XOC = True

# Specify the path to the model created with the Tensorflow franework
if XOC: model_path = '<path_to_model>'
#else: model_path = '/workspace/model_zoo/model-list/tf2_cnn_ctfar10_2.5/'
tf_model_name = 'cifar10_bw_model2.h5'
tfq_model_name = 'cifar10_bw_quant.h5'
tf_model_path = model_path + tf_model_name 
tfq_model_path = model_path + tfq_model_name

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input

# # Define a new input layer with compatible arguments
# inputs = Input(shape=(32, 32, 3))  # Replace this shape with the correct one
# x = tf_model(inputs)
# new_model = Model(inputs=inputs, outputs=x)
# new_model.save(tfq_model_path)
tf_model = tf.keras.models.load_model(tf_model_path, compile=False)
tf_model.summary()



def load_cifar10_from_directory(directory):
    # Load the training and test batches
    def load_batch(batch_file):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        return images, labels

    # Load training data
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(directory, f"data_batch_{i}")
        images, labels = load_batch(batch_file)
        x_train.append(images)
        y_train.append(labels)
    
    # Convert list of arrays into a single NumPy array
    x_train = np.concatenate(x_train, axis=0)  # Concatenate along axis 0 (vertical)
    y_train = np.concatenate(y_train, axis=0)

    # Load test data
    x_test, y_test = load_batch(os.path.join(directory, "test_batch"))
    
    # Reshape the data to (num_samples, 32, 32, 3)
    x_train = x_train.reshape(x_train.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(x_test.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(y_test)
    # Return as NumPy arrays
    return (x_train, y_train), (x_test, y_test)

# Specify the path to the extracted CIFAR-10 dataset
directory = 'cifar-10-python/cifar-10-batches-py/'


(x_train, y_train), (x_test, y_test) = load_cifar10_from_directory(directory)
# print(f"x_train shape: {x_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"x_test shape: {x_test.shape}")
# print(f"y_test shape: {y_test.shape}")
print ("Original shape:", x_train.shape, y_train.shape, x_test.shape, y_test.shape) 
x_train_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_train])
x_test_gray = np.array([cv2. cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x_test])
# Rescale grayscale image pixel range from [O:255] to [0:1]
x_train_gray_norm = (x_train_gray / 255.0).astype('float32')
x_test_gray_norm = (x_test_gray / 255.0).astype('float32')
x_train_gray_norm = x_train_gray_norm.reshape(x_train_gray_norm.shape[0], x_train_gray_norm.shape[1], x_train_gray_norm.shape[2], 1)
x_test_gray_norm =  x_test_gray_norm.reshape(x_test_gray_norm.shape[0], x_test_gray_norm.shape[1], x_test_gray_norm.shape[2], 1)
print("New shape:", x_train_gray_norm.shape, y_train.shape, x_test_gray_norm.shape, y_test.shape)
print("Type:", x_train_gray_norm.dtype, y_train.dtype, x_test_gray_norm.dtype, y_test.dtype)
print("original Label (y):", y_train[0])

y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32),depth=10)

print("new labeling(y):",y_train[0])


import pydot
import graphviz
import os
os.environ["PATH"] += ":/graphviz/bin/dot"

from tensorflow_model_optimization.quantization.keras import vitis_inspect

input_shape = (32,32,1)

target = 'DPUCZDX8G_ISA1_B4096'

inspector = vitis_inspect.VitisInspector(target=target)

inspector.inspect_model(tf_model,
                            input_shape=input_shape,
                            dump_model=True,
                            dump_model_file="inspect_model.h5",
                            # plot=True,
                            # plot_file="model.svg",
                            dump_results=True,
                            dump_results_file="inspect_results.txt",
                            verbose=0)

from tensorflow_model_optimization.quantization.keras import vitis_quantize

quantizer = vitis_quantize.VitisQuantizer(tf_model)

tfq_model = quantizer.quantize_model(
    calib_dataset=x_train_gray_norm[0:100]
)

tfq_model.save(tfq_model_path)

learning_rate= 0.0001
momentum=0
epsilon = 1e-08
batch_size_test = 1000
tfq_model =tf.keras.models.load_model(tfq_model_path)
tfq_model.compile(
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        momentum=momentum,
        epsilon=epsilon
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    metrics=['acc']
    )
tfq_model.evaluate(x_test_gray_norm,y_test,batch_size = batch_size_test)

quantizer.dump_model(tfq_model, dataset=x_train_gray_norm[0:1], dump_float=True)

