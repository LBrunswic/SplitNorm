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

T = time.time()
EPOCHS = 150
DISTRIBUTION_DIM = 2
# CHANNEL_DIM = 1
COMMAND_DIM = 0
CHANNEL_SAMPLE = 2
SAVE_FOLDER = os.path.join('results',date_time_str)
os.makedirs(SAVE_FOLDER)
model_folder = os.path.join(SAVE_FOLDER,'model')
os.makedirs(model_folder)
ARCH = 0


switch_arch = [
    [
        ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense,
        {
            'dim' : DISTRIBUTION_DIM,
            'ensemble_size' : 2,
            'n_switch' : 1,
            'inward_depth' : 4,
            'inward_width' : 8,
            'switch_dim' : 8,
            'kernelKWarg' : {'activation': tf.keras.activations.tanh},
        }
    ],
    [
        ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense,
         {
            'dim' : DISTRIBUTION_DIM,
            'ensemble_size' : 2,
            'n_switch' : 3,
            'inward_depth' : 1,
            'inward_width' : 16,
            'switch_dim' : 16,
            'kernelKWarg' : {'activation': tf.keras.activations.tanh},
        }
    ]
]

ffjord_core = ConvolutionalKernel.utils.build(switch_arch[ARCH])
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
channeller_archs = [
    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential_finite,
        {
            'distribution_dim': DISTRIBUTION_DIM,
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
            'width': 8,
            'depth': 3,
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
            'finite_set': ConvolutionalKernel.utils.switch_commands(switch_arch[1][1]['ensemble_size'],switch_arch[1][1]['n_switch'])
        }
    ]
]

conv_kernel_arch = {
    'kernel_ensemble':kernel_ensemble,
    'commander': ConvolutionalKernel.CommanderConstructors.commander_passthrough(**commander_arch1),
    'channeller': ConvolutionalKernel.utils.build(channeller_archs[ARCH]),
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
    dot_img_file=os.path.join(model_folder,'%s_graph.png' % model.name)
    tf.keras.utils.plot_model(
        model, to_file=dot_img_file, show_shapes=True,expand_nested=True,
        show_layer_activations=True,
    )
    with open(os.path.join(model_folder,'%s_summary.txt' % model.name),'w') as f:
        model.summary(print_fn=lambda x:f.write(x+'\n'))


base_distributionKWarg = {
        'loc' : tf.zeros(DISTRIBUTION_DIM),
         'scale_diag' : tf.ones(DISTRIBUTION_DIM)
    }
base_distribution = tfd.MultivariateNormalDiag(**base_distributionKWarg)



transformed_distribution = ConvKernel.build_dist(base_distribution)

# transformed_distribution = kernel_ensemble.build_dist(base_distribution)
mnist_list=[1,3,5,7,2,35,13,15,17,18,19]
batch_size = 28*28
dataset_size = 28*28
for j in mnist_list:
    image = tf.keras.datasets.mnist.load_data()[0][0][j]
    xmax, ymax = image.shape
    commands = tf.ones((*image.shape,transformed_distribution.command_dim))
    image = image.reshape(*image.shape,1)
    limits = [(-2,2,4/ymax),(-2,2,4/xmax)]
    sample = np.mgrid[[slice(a, b, e) for a, b, e in limits]].transpose().astype('float32')
    dataset_as_tensor = np.concatenate([image,sample,commands],axis=-1).reshape(xmax*ymax,-1)
    plt.matshow(tf.reshape(dataset_as_tensor[:,0],(xmax,ymax)))

# batch_size=2**8
# dataset_size = 2**14;dataset_size
# weights = tf.ones((dataset_size,1))
# moons = 1.1*(skd.make_moons(n_samples=dataset_size, noise=.06)[0].astype('float32')+np.array([[-0.5,0]],dtype='float32'))
# commands = tf.ones((dataset_size,transformed_distribution.command_dim))
# dataset_as_tensor = tf.concat([weights, moons,commands],axis=1)
# dataset_as_tensor.shape
# plt.scatter(moons[:,0],moons[:,1])



    plt.savefig(os.path.join(SAVE_FOLDER,'target_%s.png' % j))
    dataset = tf.data.Dataset.from_tensor_slices(dataset_as_tensor)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(dataset_size)
    dataset = dataset.batch(batch_size)
    transformed_distribution.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2)
    )

    #
    #
    # a = dataset_as_tensor.reshape(xmax,ymax,-1)
    # a[18,27]



    # transformed_distribution.score(moons[:batch_size],commands[:batch_size])
    for epoch in range(EPOCHS):
        i=0
        for batch in dataset:
            L = time.time()
            transformed_distribution.train_step(batch)
            i+=1
            print("epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-L))
        if epoch%10 == 0:
            transformed_distribution.display_density(name=os.path.join(SAVE_FOLDER,'epoch_%03d_number_%02d.png' % (j,epoch)))


    print("done in %s seconds" % (time.time()-T))
