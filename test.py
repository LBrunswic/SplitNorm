import tensorflow as tf;
import tensorflow_probability as tfp;
import ConvolutionalKernel;
import numpy as np
import mpl_scatter_density
import matplotlib.pyplot as plt
import sklearn.datasets as skd
import time, os, datetime
tfd = tfp.distributions
now = datetime.datetime.now()
date_time_str = now.strftime("%m-%d_%Hh%Mm%Ss")

DISTRIBUTION_DIM = 2
# CHANNEL_DIM = 1
COMMAND_DIM = 0
CHANNEL_SAMPLE = 2
SAVE_FOLDER = os.path.join('results',date_time_str)
os.makedirs(SAVE_FOLDER)

switch_arch1 = {
    'dim' : DISTRIBUTION_DIM,
    'ensemble_size' : 2,
    'n_switch' : 1,
    'inward_depth' : 3,
    'inward_width' : 16,
    'switch_dim' : 8,
    'kernelKWarg' : {'activation': tf.keras.activations.tanh},
}

ffjord_core = ConvolutionalKernel.ModelConstructors.commanded_switched_ensemble_sequential_dense(**switch_arch1)
infra_command = ffjord_core.command_dim


FFJORD_dorpri_arch1 = {
    'state_time_derivative_fn': ffjord_core,
    'ode_solve_fn': tfp.math.ode.DormandPrince(atol=1e-5).solve,
    'trace_augmentation_fn': tfp.bijectors.ffjord.trace_jacobian_exact
}
kernel_ensemble_arch = {
    'flow_family': tfp.bijectors.FFJORD(**FFJORD_dorpri_arch1),
    'command_dim': infra_command,
    'distribution_dim': DISTRIBUTION_DIM
}
kernel_ensemble = ConvolutionalKernel.FlowEnsemble(**kernel_ensemble_arch)

commander_arch1 = {
    'channel_dim' : infra_command,
    'command_dim' : COMMAND_DIM,
    'output_dim' : infra_command,
}
channeller_arch1 = {
    'distribution_dim': DISTRIBUTION_DIM,
    'channel_dim': infra_command,
    'command_dim': COMMAND_DIM,
    'width': 1,
    'depth': 1,
    'kernelKWarg': {'activation': tf.keras.layers.LeakyReLU()},
    # 'final_activation' : tf.keras.layers.Activation('softmax')
    'finite_set': tf.eye(infra_command)
}

conv_kernel_arch = {
    'kernel_ensemble':kernel_ensemble,
    'commander': ConvolutionalKernel.ModelConstructors.commander_passthrough(**commander_arch1),
    # 'channeller': ConvolutionalKernel.ModelConstructors.channeller_sequential(**channeller_arch1),
    'channeller': ConvolutionalKernel.ModelConstructors.channeller_sequential_finite(**channeller_arch1),
    'channel_dim': infra_command,
    'command_dim': COMMAND_DIM,
}
ConvKernel = ConvolutionalKernel.ConvKernel(**conv_kernel_arch)


models = [
    ffjord_core.kernel,
    ConvKernel.channeller,
    ConvKernel.commander
]
for model in models:
    dot_img_file=os.path.join(SAVE_FOLDER,'%s.png' % model.name)
    tf.keras.utils.plot_model(
        model, to_file=dot_img_file, show_shapes=True,expand_nested=True,
        show_layer_activations=True,
    )


base_distributionKWarg = {
        'loc' : tf.zeros(DISTRIBUTION_DIM),
         'scale_diag' : tf.ones(DISTRIBUTION_DIM)
    }
base_distribution = tfd.MultivariateNormalDiag(**base_distributionKWarg)

batch_size=2**8
dataset_size = 2**14;dataset_size

transformed_distribution = ConvKernel.build_dist(base_distribution)
# transformed_distribution = kernel_ensemble.build_dist(base_distribution)

weights = tf.ones((dataset_size,1))
moons = 1.1*(skd.make_moons(n_samples=dataset_size, noise=.06)[0].astype('float32')+np.array([[-0.5,0]],dtype='float32'))
commands = tf.ones((dataset_size,transformed_distribution.command_dim))
dataset_as_tensor = tf.concat([weights, moons,commands],axis=1)
dataset_as_tensor.shape
dataset = tf.data.Dataset.from_tensor_slices(dataset_as_tensor)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
# dataset = dataset.shuffle(dataset_size)
dataset = dataset.batch(batch_size)
transformed_distribution.compile(
    optimizer=tf.keras.optimizers.Adam(1e-2)
)

plt.scatter(moons[:,0],moons[:,1])
plt.savefig(os.path.join(SAVE_FOLDER,'target.png'))
for epoch in range(1000):
    # transformed_distribution.fit(
    #     dataset,
    #     epochs=1
    # )
    for batch in dataset:
        transformed_distribution.train_step(batch)
    transformed_distribution.display_density(name=os.path.join(SAVE_FOLDER,'epoch_%s.png' % epoch))


#
# self = transformed_distribution
# weighted_sample_command_batch=dataset[:64]
# weight_batch, sample_batch, command_batch = tf.split(
#     weighted_sample_command_batch,[1,self.distribution_dim,-1],
#     axis=-1
# )
# weight_batch=tf.reshape(weight_batch,(-1,))
#
# command_batch.shape
# sample_batch.shape
# trainable_var = self.trainable_variables
# with tf.GradientTape() as tape:
#     scores = self.score(sample_batch,command_batch)
#     scores = weight_batch*scores
#     loss = tf.reduce_mean(scores)
# grad = tape.gradient(loss, trainable_var)
# self.optimizer.apply_gradients(zip(grad, trainable_var))
#
#
# weighted_channel_batch = ConvKernel.channeller((sample_batch, command_batch))
# weighted_channel_batch.shape
# channel_batch = tf.unstack(weighted_channel_batch[:, :, :-1], axis=1)[0]
# channel_batch.shape
# transformed_distribution = ConvKernel.kernel_ensemble.build_dist(base_distribution)
# commander_out =  ConvKernel.commander((channel_batch, command_batch))
# commander_out.shape
# sample_batch.shape
# transformed_distribution.score(sample_batch, commander_out)
# transformed_distribution.parametrized_transformed_distribution.log_prob(
#     sample_batch,
#     bijector_kwargs={'command':commander_out}
# )
# L = tf.keras.layers.Dot(axes=(-1))
# L([tf.ones((64,2,2)),tf.ones((64,2))]).shape
