import tensorflow as tf
import itertools

def _aux_sequential_dense_gen(out_dim, hidden_width, depth, kernelKWarg=None):
    if kernelKWarg is None:
        kernelKWarg = {}
    layers = [
        tf.keras.layers.Dense(hidden_width,**kernelKWarg)
        for i in range(depth-1)
    ]
    layers+=[tf.keras.layers.Dense(out_dim, **kernelKWarg)]
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
