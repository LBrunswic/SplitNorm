import tensorflow as tf
from ConvolutionalKernel.utils import _aux_sequential_dense_gen

def channeller_sequential(distribution_dim=None,channel_dim=None,channel_sample=1,command_dim=None,width=8,depth=3,kernelKWarg=None,final_activation=None):
    """ Channeller model constructor. """
    distribution_input = tf.keras.layers.Input(shape=(distribution_dim,))
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    core_model = _aux_sequential_dense_gen((channel_dim+1)*channel_sample,width,depth,kernelKWarg=kernelKWarg)
    raw_outputs = core_model(tf.keras.layers.Concatenate()([distribution_input,command_input]))
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
