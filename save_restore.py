import tensorflow as tf
import keras
import os

mnist = keras.datasets.mnist
batch_size = 128
num_class = 10
epoch = 10

(train_data, train_label), (test_data, test_label) = mnist.load_data()
