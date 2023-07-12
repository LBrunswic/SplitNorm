import tensorflow as tf
from ConvolutionalKernel.utils import _aux_sequential_dense_gen

def commander_sequential(channel_dim=None, command_dim=None, output_dim=None, depth=1, width=8, kernel_kwargs = {}):
    """Commander model constructor. Concatenate command and channel and pass to a Sequential model"""
    layers = [tf.keras.layers.Input(shape=(channel_dim+command_dim,))]
    layers += [
        tf.keras.layers.Dense(width,**kernel_kwargs)
        for i in range(depth-1)
    ]
    layers+=[tf.keras.layers.Dense(output_dim, **kernel_kwargs)]
    core_model = tf.keras.Sequential(layers)

    channel_input = tf.keras.layers.Input(shape=(channel_dim,))
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (channel_input,command_input)
    outputs = core_model(tf.keras.layers.Concatenate()([channel_input,command_input]))
    return tf.keras.Model(inputs=inputs, outputs=outputs,name='commander')

def commander_passthrough(channel_dim=None,command_dim=None,output_dim=None):
    """ Commander model constructor. Concatenate channel and command"""
    channel_input = tf.keras.layers.Input(shape=(channel_dim,))
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (channel_input,command_input)
    outputs = tf.keras.layers.Concatenate()([channel_input,command_input])
    return tf.keras.Model(inputs=inputs,outputs=outputs,name='commander')
