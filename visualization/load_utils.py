import pandas
import numpy as np
import os
import tensorflow as tf
from datetime import datetime
def load_MNIST():
    images = np.load('MNIST.npy')
    labels = np.load('MNIST_labels.npy')
    return images, labels

class evaluate(tf.keras.callbacks.Callback):
    """Errors computation and record
    """

    def __init__(self, x_test, y_test, hparams):
        self.best = 1000
        self.y_test = y_test
        self.x_test = x_test
        self.res = pandas.DataFrame({})
        self.hparams = hparams

    def on_epoch_end(self, epoch, logs=None):
        res = self.model(self.x_test)
        if len(res.shape) == 3:
            res = res[..., 0]
        prediction = tf.argmax(res, axis=1)
        current = tf.reduce_sum(tf.cast(self.y_test != prediction, 'int32')).numpy()
        self.best = min(self.best, current)

        to_pandas = {**self.hparams}
        to_pandas.update({
            'epoch': epoch,
            'errors': current
        })
        to_pandas = pandas.DataFrame(to_pandas, [epoch])
        self.res = pandas.concat([self.res, to_pandas])

    def on_train_end(self, logs=None):
        os.makedirs(self.hparams['log_dir'])
        self.res.to_csv(os.path.join(self.hparams['log_dir'], 'results.csv'))
        pass

def lr_schedule_gen(hparama):
    def lr_schedule(epoch, lr):
        return hparama['LR'] * tf.cast(tf.math.cos(np.pi / 2 * (epoch % hparama['EPOCHS']) / hparama['EPOCHS']), 'float32')
    return lr_schedule

def raw_mnist_model_gen(hparams):
    model_raw = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(28, 28))] +
        [tf.keras.layers.Dense(hparams['W'], activation=hparams['activation']) for _ in range(hparams['L'])] +
        [tf.keras.layers.Dense(10)]
    )
    return model_raw
def channel_mnist_model_gen(hparams):
    model_raw = tf.keras.models.Sequential(
        [tf.keras.layers.Flatten(input_shape=(21,))] +
        [tf.keras.layers.Dense(hparams['W'], activation=hparams['activation']) for _ in range(hparams['L'])] +
        [tf.keras.layers.Dense(10)]
    )
    return model_raw

def train_mnist_raw(hparams,split=50000,lr_schedule_gen=lr_schedule_gen):
    model_raw = raw_mnist_model_gen(hparams)
    model_raw.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['LR']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model_raw.fit(
        x_train[:split], y_train[:split],
        batch_size=hparams['BATCH_SIZE'],
        validation_split=0.1,
        verbose=0,
        epochs=hparams['EPOCHS'] ,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule_gen(hparams))],
    )
    return model_raw.evaluate(x_train[split:], y_train[split:])

def train_mnist_channel(hparams,channel,split=50000,lr_schedule_gen=lr_schedule_gen):
    model_channel = channel_mnist_model_gen(hparams)
    model_channel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams['LR']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model_channel.fit(
        channel[:split], y_train[:split],
        batch_size=hparams['BATCH_SIZE'],
        epochs=hparams['EPOCHS'] ,
        verbose=0,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule_gen(hparams))],
    )
    return model_channel.evaluate(channel[split:],y_train[split:])


if __name__ == '__main__':

    hparams = {
        'W': 16,
        'L': 3,
        'LR': 0.01,
        'BATCH_SIZE': 512,
        'EPOCHS': 10,
        'activation':'tanh'
    }
    CHANNELS_FOLDER = os.path.join('results','10-30_08h11m51s/')





    print('Here are some vizualizations')
    x_train, y_train = load_MNIST()
    last_channel = len([x for x in os.listdir(CHANNELS_FOLDER) if 'channel' in x])
    channels = np.stack([np.load(os.path.join(CHANNELS_FOLDER,'channel_dist_%s' % i)) for i in range(last_channel)])

    hparams.update({'log_dir':os.path.join(CHANNELS_FOLDER,'raw_classification.csv')})
    print(train_mnist_raw(hparams))


    for i,channel in enumerate(channels):
        hparams.update({'log_dir': os.path.join(CHANNELS_FOLDER, 'channel%s_classification.csv' % i)})
        print(train_mnist_channel(hparams,channel[:,0]))



