import tensorflow as tf

channeller_archs = [
    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
        {
            'distribution_dim': DISTRIBUTION_DIM,
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
            'width': 1,
            'depth': 1,
            'kernelKWarg': {'activation': tf.keras.layers.LeakyReLU()},
            # 'final_activation' : tf.keras.layers.Activation('softmax')
            'finite_set': tf.eye(infra_command)
        }
    ],
    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential_finite,
        {
            'distribution_dim': DISTRIBUTION_DIM,
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
            'width': 1,
            'depth': 1,
            'kernelKWarg': {'activation': tf.keras.layers.LeakyReLU()},
            # 'final_activation' : tf.keras.layers.Activation('softmax')
            'finite_set': ConvolutionalKernel.utils.switch_commands(switch_arch2['ensemble_size'],switch_arch2['n_switch'])
        }
    ]
]
