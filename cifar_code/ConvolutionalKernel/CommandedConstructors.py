import tensorflow as tf
import ConvolutionalKernel.CommandedModels as CMLP
from ConvolutionalKernel.utils import _aux_sequential_dense_gen

def commanded_switched_ensemble_sequential_dense(dim=2,n_switch=4,ensemble_size=2,inward_depth=1,switch_dim=16,inward_width=8,kernelKWarg={}):
    """ Commanded ODE MLP : switch architecture"""
    command_dim = n_switch * ensemble_size
    input_x = tf.keras.Input(shape=(dim+1,))
    input_command = tf.keras.Input(shape=(command_dim,))
    inputs = (input_x,input_command)
    switch_commands = tf.split(input_command, n_switch, axis=1)
    flow_family = [
        [_aux_sequential_dense_gen(switch_dim,inward_width,inward_depth,kernelKWarg=kernelKWarg) for _ in range(ensemble_size)]
         for _ in range(n_switch-1)
    ]
    flow_family += [[_aux_sequential_dense_gen(dim,inward_width,inward_depth,kernelKWarg=kernelKWarg) for _ in range(ensemble_size)]]
    level_out = input_x
    for level in range(n_switch):
        flow_level_out = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Reshape((-1,1))(F(level_out)) for F in flow_family[level]])
        # (batch_size, switch_dim, channel_size)
        level_out = tf.keras.layers.Dot(axes=-1)([
            flow_level_out,
            switch_commands[level]
        ]) #(batch_size, data_dim)
    ode_fn = tf.keras.Model(inputs=inputs, outputs=level_out,name='seq_flow')
    return CMLP.CommandedMLP_ODE(ode_fn,command_dim,name='ODE_MLP')

import math as m
def torus_positional_encoding(dim=2,cutoff=8,wx=4):
    pi = tf.constant(m.pi)
    input_x = tf.keras.Input(shape=(dim,))
    if cutoff == 0:
        outputs = input_x[:,:0]
    else:
        multi_x = tf.keras.layers.Concatenate()([float(i)*input_x for i in tf.range(1,cutoff+1)])
        outputs = tf.keras.layers.Concatenate()([tf.math.cos(2*pi/wx*multi_x),tf.math.sin(multi_x)])
    return tf.keras.Model(inputs=input_x, outputs=outputs,name='positional_encoding_k%s' % cutoff)


def commanded_switched_ensemble_sequential_dense_with_encoding(dim=2, add_x=True, n_switch=4,cutoff = 8, ensemble_size=2,inward_depth=1,switch_dim=16,inward_width=8,kernelKWarg={},final_activation=lambda x:x):
    command_dim = n_switch * ensemble_size
    input_x = tf.keras.Input(shape=(dim+1,))
    input_command = tf.keras.Input(shape=(command_dim,))
    inputs = (input_x,input_command)

    encoding = torus_positional_encoding(dim=dim,cutoff=cutoff)
    x,t = tf.split(input_x,[dim,1],axis=-1)
    if add_x:
        input_encoded_x =  tf.keras.layers.Concatenate()([x,encoding(x),t])
    else:
        input_encoded_x =  tf.keras.layers.Concatenate()([encoding(x),t])
    switch_commands = tf.split(input_command, n_switch, axis=1)
    flow_family = [
        [_aux_sequential_dense_gen(switch_dim,inward_width,inward_depth,kernelKWarg=kernelKWarg) for _ in range(ensemble_size)]
         for _ in range(n_switch-1)
    ]
    flow_family += [[_aux_sequential_dense_gen(dim,inward_width,inward_depth,kernelKWarg=kernelKWarg) for _ in range(ensemble_size)]]
    level_out = input_encoded_x
    for level in range(n_switch):
        flow_level_out = tf.keras.layers.Concatenate(axis=-1)([tf.keras.layers.Reshape((-1,1))(F(level_out)) for F in flow_family[level]])
        # (batch_size, switch_dim, channel_size)
        level_out = tf.keras.layers.Dot(axes=-1)([
            flow_level_out,
            switch_commands[level]
        ]) #(batch_size, data_dim)
    ode_fn = tf.keras.Model(inputs=inputs, outputs=final_activation(level_out),name='seq_flow')
    return CMLP.CommandedMLP_ODE(ode_fn,command_dim,name='ODE_MLP')


def commanded_concat_sequential_dense_with_encoding(dim=2, add_x=True, cutoff = 8,command_bypass=True,time_bypass=True, command_dim=1, depth=1,width=8,kernelKWarg={},final_activation=lambda x:x):
    input_x = tf.keras.Input(shape=(dim+1,))
    input_command = tf.keras.Input(shape=(command_dim,))
    inputs = (input_x,input_command)

    encoding = torus_positional_encoding(dim=dim,cutoff=cutoff)
    x,t = tf.split(input_x,[dim,1],axis=-1)
    if add_x:
        input_encoded_x = tf.keras.layers.Concatenate()([x,encoding(x),input_command,t])
    else:
        input_encoded_x = tf.keras.layers.Concatenate()([encoding(x),input_command,t])
    level_out = input_encoded_x
    for level in range(depth-1):
        level_out = tf.keras.layers.Dense(width,**kernelKWarg)(level_out)
        level_out = tf.keras.layers.Concatenate()([
            level_out]+[input_command]*command_bypass+[t]*time_bypass)
    level_out = tf.keras.layers.Dense(dim,**kernelKWarg)(level_out)
    ode_fn = tf.keras.Model(inputs=inputs, outputs=final_activation(level_out),name='seq_flow')
    return CMLP.CommandedMLP_ODE(ode_fn,command_dim,name='ODE_MLP')
