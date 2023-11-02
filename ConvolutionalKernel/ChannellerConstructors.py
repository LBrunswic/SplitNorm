import tensorflow as tf
from ConvolutionalKernel.utils import _aux_sequential_dense_gen,_aux_sequential_conv_gen

def channeller_sequential(
    distribution_dim=None,
    distribution_shape=None,
    channel_dim=None,
    channel_sample=1,
    command_dim=None,
    width=8,
    depth=3,
    kernelKWarg=None,
    final_activation=lambda x:x,
    flatten=True,
    keep=...,
    weights_moderation = lambda x:x,
    final_rescale = 1e-3,
    name="channeller",
):
    """ Channeller model constructor. """
    if distribution_shape is None:
        distribution_shape = (distribution_dim,)
    distribution_input = tf.keras.layers.Input(shape=distribution_shape)
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    core_model = _aux_sequential_dense_gen((channel_dim+1)*channel_sample,width,depth,kernelKWarg=kernelKWarg)
    if flatten:
        pre1 = tf.keras.layers.Flatten()
        pre2 = tf.keras.layers.Flatten()
    else:
        pre1 = lambda x:x
        pre2 = lambda x:x
    raw_outputs = core_model(tf.keras.layers.Concatenate()([pre1(distribution_input[keep]),pre2(command_input)]))
    channel_batch, weights = tf.split(tf.keras.layers.Reshape((channel_sample,channel_dim+1))(raw_outputs),[channel_dim,1],axis=-1)
    outputs = tf.keras.layers.Concatenate(axis=-1)([final_activation(final_rescale*channel_batch),tf.nn.softmax(weights_moderation(weights),axis=1)])

    return tf.keras.Model(inputs=inputs, outputs=outputs,name=name)

def channeller_sequential_finite(
    finite_set=None,
    distribution_dim=None,
    distribution_shape=None,
    channel_dim=None,
    command_dim=None,
    width=8,
    depth=3,
    kernelKWarg=None,
    final_activation=lambda x:x,
    flatten=True,
    keep=...,
    weights_moderation = lambda x:x,
    name="channeller",
):
    """ Channeller model constructor. Outputing a distribution on a finite subset of the channel space"""
    assert(len(finite_set.shape)==2)
    assert(finite_set.shape[0]>0)
    assert(finite_set.shape[1]==channel_dim)
    if distribution_shape is None:
        distribution_shape = (distribution_dim,)
    distribution_input = tf.keras.layers.Input(shape=distribution_shape)
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    channel_sample = finite_set.shape[0]

    core_model = _aux_sequential_dense_gen(channel_sample,width,depth,kernelKWarg=kernelKWarg)
    weights = tf.keras.layers.Reshape((-1,1))(core_model(tf.keras.layers.Concatenate()([distribution_input,command_input])))
    bc_finite_set = tf.broadcast_to(finite_set,tf.where([True,True,False],tf.shape(weights),[0,0,channel_dim]))
    outputs = tf.keras.layers.Concatenate(axis=-1)([bc_finite_set,tf.keras.layers.Activation('softmax')(weights)])

    return tf.keras.Model(inputs=inputs, outputs=outputs,name=name)

def channeller_trivial(
    distribution_dim=None,
    distribution_shape=None,
    channel_dim=None,
    command_dim=None,
    name="channeller",
):
    if distribution_shape is None:
        distribution_shape = (distribution_dim,)
    distribution_input = tf.keras.layers.Input(shape=distribution_shape)
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    outputs = tf.broadcast_to(tf.ones((1,1,channel_dim+1)),tf.where([True,False,False],tf.shape(distribution_input),[0,1,channel_dim+1]))
    return tf.keras.Model(inputs=inputs, outputs=outputs,name=name)



def channeller_sequential_conv(
    distribution_dim=None,
    distribution_shape=None,
    channel_dim=None,
    channel_sample=1,
    command_dim=None,
    width=8,
    depth=3,
    kernelKWarg=None,
    final_activation=lambda x:x,
    flatten=True,
    keep=...,
    weights_moderation = lambda x:x,
    final_rescale = 1e-3
):
    """ Channeller model constructor. """
    if distribution_shape is None:
        distribution_shape = (distribution_dim,)

    distribution_input = tf.keras.layers.Input(shape=distribution_shape)
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    core_model = _aux_sequential_conv_gen((channel_dim+1)*channel_sample,width,depth,kernelKWarg=kernelKWarg)
    # if flatten:
    #     pre1 = tf.keras.layers.Flatten()

    #     pre2 = tf.keras.layers.Flatten()
    # else:
    #     pre1 = lambda x:x
    #     pre2 = lambda x:x
    # raw_outputs = core_model(tf.keras.layers.Concatenate()([distribution_input[keep],command_input]))
    # raw_outputs = core_model(tf.expand_dims(tf.reshape(distribution_input[keep], shape=[-1,28,28,3]), axis=0))
    raw_outputs = core_model(tf.reshape(distribution_input[keep], shape=[-1,32,32,3]))
    channel_batch, weights = tf.split(tf.keras.layers.Reshape((channel_sample,channel_dim+1))(raw_outputs),[channel_dim,1],axis=-1)
    outputs = tf.keras.layers.Concatenate(axis=-1)([final_activation(final_rescale*channel_batch),tf.nn.softmax(weights_moderation(weights),axis=1)])

    return tf.keras.Model(inputs=inputs, outputs=outputs,name="channeller")


def channeller_sequential_resnet(
    distribution_dim=None,
    distribution_shape=None,
    channel_dim=None,
    channel_sample=1,
    command_dim=None,
    width=8,
    depth=3,
    kernelKWarg=None,
    final_activation=lambda x:x,
    flatten=True,
    keep=...,
    weights_moderation = lambda x:x,
    final_rescale = 1e-3
):
    """ Channeller model constructor. """
    if distribution_shape is None:
        distribution_shape = (distribution_dim,)

    distribution_input = tf.keras.layers.Input(shape=distribution_shape)
    command_input = tf.keras.layers.Input(shape=(command_dim,))
    inputs = (distribution_input,command_input)
    core_model = _aux_sequential_conv_gen((channel_dim+1)*channel_sample,width,depth,kernelKWarg=kernelKWarg)
    # if flatten:
    #     pre1 = tf.keras.layers.Flatten()

    #     pre2 = tf.keras.layers.Flatten()
    # else:
    #     pre1 = lambda x:x
    #     pre2 = lambda x:x
    # raw_outputs = core_model(tf.keras.layers.Concatenate()([distribution_input[keep],command_input]))
    # raw_outputs = core_model(tf.expand_dims(tf.reshape(distribution_input[keep], shape=[-1,28,28,3]), axis=0))
    raw_outputs = core_model(tf.reshape(distribution_input[keep], shape=[-1,32,32,3]))
    channel_batch, weights = tf.split(tf.keras.layers.Reshape((channel_sample,channel_dim+1))(raw_outputs),[channel_dim,1],axis=-1)
    outputs = tf.keras.layers.Concatenate(axis=-1)([final_activation(final_rescale*channel_batch),tf.nn.softmax(weights_moderation(weights),axis=1)])

    return tf.keras.Model(inputs=inputs, outputs=outputs,name="channeller")



class InceptionBase(tf.keras.layers.Layer):
    def __init__(self, structure=None, pool_struc=None, kernel_options=None, strides=2,  name='InceptionLayer', **kwargs):
        super(InceptionBase,self).__init__(name=name,**kwargs)
        self.structure = structure
        self.kernel_options = kernel_options
        self.pool_struc = pool_struc
        self.strides = strides


    def build(self,input_shape):
        inputs = tf.keras.layers.Input(input_shape[1:])
        pool_line = tf.keras.layers.MaxPooling2D(
            pool_size=self.pool_struc['pool_size'],
            strides=1,
            padding=self.pool_struc['padding']
        )(inputs)
        lines = [inputs] * len(self.structure[:-1]) + [pool_line]

        for line in range(len(self.structure)):
            for i,stage in enumerate(self.structure[line][:-1]):
                lines[line] = tf.keras.layers.Conv2D(
                    stage[0],
                    kernel_size=stage[1],
                    strides=1,
                    **self.kernel_options
                )(lines[line])
            stage = self.structure[line][-1]
            lines[line] = tf.keras.layers.Conv2D(
                stage[0],
                kernel_size=stage[1],
                strides=self.strides,
                **self.kernel_options
            )(lines[line])

        outputs = tf.keras.layers.Concatenate(axis=-1)(lines)
        self.layer = tf.keras.models.Model(inputs=inputs,outputs=outputs)

    def call(self,inputs, training=False):
        return self.layer(inputs, training=training)


