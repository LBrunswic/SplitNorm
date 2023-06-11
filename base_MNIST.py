# This basic MNIST test is inteneded to ensurethat given architecture for the channeller have gather
# have th e ability to fit MNIST.

import tensorflow as tf;
import ConvolutionalKernel;
import numpy as np
import sys
LR = 5*1e-3
DEPTH = int(sys.argv[1])
WIDTH = int(sys.argv[2])
EPOCHS = int(sys.argv[3])
(x,y),(z,t) = tf.keras.datasets.mnist.load_data()
y = y.astype('int16')
y = np.eye(10)[y]
t = np.eye(10)[t]
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28))] + [
    tf.keras.layers.Dense(WIDTH, activation=tf.keras.layers.LeakyReLU())
    for _ in range(DEPTH-1)
]+
[tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)]
)
model.compile(
    optimizer=tf.optimizers.Adam(LR),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
model.fit(x,y,epochs=EPOCHS,use_multiprocessing=True)
model.evaluate(z,t)

model.compile(
    optimizer=tf.optimizers.Adam(LR/10),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)
model.fit(x,y,epochs=EPOCHS,use_multiprocessing=True)
model.evaluate(z,t)
