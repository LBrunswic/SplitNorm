import tensorflow as tf
from ConvolutionalKernel.utils import _aux_sequential_dense_gen

def channeller_sequential(
    distribution_dim=None,
    distribution_shape=None,
    channel_dim=None,
    channel_sample=1,
    command_dim=None,
    width=8,
    depth=3,
    kernelKWarg=None,
    final_activation=None,
    flatten=True,
    keep=...,
):
    """ Channeller model constructor. """
    if distribution_shape is None:
        distribution_shape = (distribution_dim,)
    distribution_input = tf.keras.layers.Input(shape=distribution_shape)
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    core_model = _aux_sequential_dense_gen((channel_dim+1)*channel_sample,width,depth,kernelKWarg=kernelKWarg)
    if flatten:
        pre = tf.keras.layers.Flatten()
    else:
        pre = lambda x:x
    raw_outputs = core_model(tf.keras.layers.Concatenate()([pre(distribution_input[keep]),pre(command_input)]))
    channel_batch, weights = tf.split(tf.keras.layers.Reshape((channel_sample,channel_dim+1))(raw_outputs),[channel_dim,1],axis=-1)
    outputs = tf.keras.layers.Concatenate(axis=-1)([channel_batch,tf.keras.layers.Activation('softmax')(weights)])
    if final_activation is not None:
        outputs = final_activation(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs,name="channeller")

def channeller_sequential_finite(finite_set=None, distribution_dim=None,channel_dim=None,command_dim=None,width=8,depth=3,kernelKWarg=None):
    """ Channeller model constructor. Outputing a distribution on a finite subset of the channel space"""
    assert(len(finite_set.shape)==2)
    assert(finite_set.shape[0]>0)
    assert(finite_set.shape[1]==channel_dim)
    distribution_input = tf.keras.layers.Input(shape=(distribution_dim,))
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    channel_sample = finite_set.shape[0]
    core_model = _aux_sequential_dense_gen(channel_sample,width,depth,kernelKWarg=kernelKWarg)
    weights = tf.keras.layers.Reshape((-1,1))(core_model(tf.keras.layers.Concatenate()([distribution_input,command_input])))
    bc_finite_set = tf.broadcast_to(finite_set,tf.where([True,True,False],tf.shape(weights),[0,0,channel_dim]))
    outputs = tf.keras.layers.Concatenate(axis=-1)([bc_finite_set,tf.keras.layers.Activation('softmax')(weights)])
    return tf.keras.Model(inputs=inputs, outputs=outputs,name="channeller")

def channeller_trivial(distribution_dim=None, channel_dim=None, command_dim=None):
    distribution_input = tf.keras.layers.Input(shape=(distribution_dim,))
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    outputs = tf.reshape(0*tf.reduce_sum(distribution_input,axis=1)+0*tf.reduce_sum(command_input,axis=1),(-1,1,1)) + tf.ones((1,1,channel_dim+1))
    return tf.keras.Model(inputs=inputs, outputs=outputs,name='channeller')
