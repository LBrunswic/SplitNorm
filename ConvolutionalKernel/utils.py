import tensorflow as tf
import itertools

def _aux_sequential_dense_gen(out_dim, hidden_width, depth, kernelKWarg=None):
    if kernelKWarg is None:
        kernelKWarg = {}
    layers = [
        tf.keras.layers.Dense(hidden_width,**kernelKWarg)
        for i in range(depth-1)
    ]
    layers+=[tf.keras.layers.Dense(
        out_dim,
        **kernelKWarg,
        # kernel_initializer=tf.keras.initializers.Constant(0.)
    )]
    return tf.keras.Sequential(layers)

def switch_commands(switch_size,n_switch):
    # channel_sample = switch_size**n_switch
    base_choices = tf.eye(switch_size)
    return tf.stack([
        tf.concat(choice ,axis=0)
        for choice in itertools.product(base_choices,repeat=n_switch)
    ])

def build(param):
    return param[0](**param[1])



def resnet_block(inputs, filters, strides=1):
    identity = inputs
    if strides==2:
        identity = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(identity)
        identity = tf.keras.layers.BatchNormalization()(identity)
        identity = tf.keras.layers.Activation('relu')(identity)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, identity])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def _aux_sequential_conv_gen(out_dim, hidden_width, depth, kernelKWarg=None):
    input_shape = (32, 32, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=64, strides=1)
    x = resnet_block(x, filters=128, strides=2)
    x = resnet_block(x, filters=128, strides=1)
    x = resnet_block(x, filters=256, strides=2)
    x = resnet_block(x, filters=256, strides=1)
    x = resnet_block(x, filters=512, strides=2)
    x = resnet_block(x, filters=512, strides=1)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(12)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model
