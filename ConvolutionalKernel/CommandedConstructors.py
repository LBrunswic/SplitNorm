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
