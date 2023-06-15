import ConvolutionalKernel
import tensorflow as tf
import numpy as np

    inward_depth = 2
    inward_width = 64
    switch_dim = inward_width
    n_switch = 3
    ensemble_size = 2
Architectures = {
# Switchable Arch with 3 switchs of width 2. Internal dense layers of width holding in Leo's laptop memory

    'MediumDense3x2_nofourier': {
          'channeller_type' : ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
          'channeller_param':  {
              'channel_sample': 3,
              'width': 32,
              'depth': 3,
          },

          'flow_type': ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense_with_encoding,
          'flow_param' : {
              'ensemble_size' : 2,
              'cutoff' : 0,
              'add_x': True,
              'n_switch' : 3,
              'inward_depth' : 2,
              'inward_width' : 64,
              'switch_dim' : 64,
          }
      ],
    },

# Switchable Arch with 3 switchs of width 2. Internal dense layers

    'Large3x2_nofourier': {
          'channeller_type' : ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
          'channeller_param':  {
              'channel_sample': 5,
              'width': 32,
              'depth': 4,
          },

          'flow_type': ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense_with_encoding,
          'flow_param' : {
              'ensemble_size' : 2,
              'cutoff' : 0,
              'add_x': True,
              'n_switch' : 3,
              'inward_depth' : 2,
              'inward_width' : 128,
              'switch_dim' : 128,
          }
      ],
    }

}
