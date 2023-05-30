import tensorflow as tf
import ConvolutionalKernel.CommandedModels as CMLP

def _aux_sequential_dense_gen(out_dim, hidden_width, depth, kernelKWarg=None):
    if kernelKWarg is None:
        kernelKWarg = {}
    layers = [
        tf.keras.layers.Dense(hidden_width,**kernelKWarg)
        for i in range(depth-1)
    ]
    layers+=[tf.keras.layers.Dense(out_dim, **kernelKWarg)]
    return tf.keras.Sequential(layers)

def commanded_switched_ensemble_sequential_dense(dim=2,n_switch=4,ensemble_size=2,inward_depth=1,switch_dim=16,inward_width=8,kernelKWarg={}):
    """ Commanded MLP made """
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

def commander_sequential(channel_dim=None, command_dim=None, output_dim=None, depth=1, width=8, kernel_kwargs = {}):
    """Commander model constructor. Concatenate command and channel and pass to a Sequential model"""
    layers = [tf.keras.layers.Input(input_shape=(channel_dim+command_dim,))]
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
