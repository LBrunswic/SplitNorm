import tensorflow as tf;
import tensorflow_probability as tfp;
import ConvolutionalKernel;
import numpy as np
import mpl_scatter_density
import matplotlib.pyplot as plt
import sklearn.datasets as skd
import time, os, datetime
import imageio.v3 as iio
from PIL import Image

tfd = tfp.distributions
now = datetime.datetime.now()
date_time_str = now.strftime("%m-%d_%Hh%Mm%Ss")
IMAGE = 'nasa_galaxy.png'

image = iio.imread(os.path.join('images', IMAGE))
image.shape


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
            'ensemble_size' : 3,
            'n_switch' : 1,
            'inward_depth' : 3,
            'inward_width' : 16,
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

ffjord_cores = [ ConvolutionalKernel.utils.build(switch_arch[ARCH]) for _ in range(3)]

infra_command = ffjord_cores[0].command_dim

FFJORD_dorpri_archs = [
    {
        'state_time_derivative_fn': ffjord_core,
        'ode_solve_fn': tfp.math.ode.DormandPrince(atol=1e-5).solve,
        'trace_augmentation_fn': tfp.bijectors.ffjord.trace_jacobian_exact
    }
    for ffjord_core in ffjord_cores
]

kernel_ensemble_archs = [
    {
        'flow_family': tfp.bijectors.FFJORD(**FFJORD_dorpri_arch),
        'command_dim': infra_command,
        'distribution_dim': DISTRIBUTION_DIM
    }
    for FFJORD_dorpri_arch in FFJORD_dorpri_archs
]
kernel_ensembles = [
    ConvolutionalKernel.FlowEnsemble(**kernel_ensemble_arch)
    for kernel_ensemble_arch in  kernel_ensemble_archs
]

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

conv_kernel_archs = [
    {
        'kernel_ensemble':kernel_ensemble,
        'commander': ConvolutionalKernel.CommanderConstructors.commander_passthrough(**commander_arch1),
        'channeller': ConvolutionalKernel.utils.build(channeller_archs[ARCH]),
        'channel_dim': infra_command,
        'command_dim': COMMAND_DIM,
    }
    for kernel_ensemble in kernel_ensembles
]
ConvKernels = [ConvolutionalKernel.ConvKernel(**conv_kernel_arch) for conv_kernel_arch in conv_kernel_archs]


models = [
    ffjord_cores[0].kernel,
    ConvKernels[0].channeller,
    ConvKernels[0].commander
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



transformed_distributions = [ConvKernel.build_dist(base_distribution) for ConvKernel in ConvKernels]

image = iio.imread(os.path.join('images', IMAGE))
image.shape
batch_size = 2**6
dataset_size = image.shape[0]*image.shape[1]
xmax, ymax,colors = image.shape
commands = tf.ones((*image.shape,transformed_distributions[0].command_dim))
image = image.reshape(*image.shape,1)
limits = [(-2,2,4/ymax),(-2,2,4/xmax)]
coordinates = np.mgrid[[slice(a, b, e) for a, b, e in limits]].transpose().astype('float32').reshape(xmax,ymax,1,2)
coordinates.shape
coordinates = tf.broadcast_to(coordinates,(xmax,ymax,3,2))
dataset_as_tensor = np.concatenate([image,coordinates,commands],axis=-1).reshape(xmax*ymax,colors,-1)
dataset_as_tensor.shape


#
# T = time.time()
# a = [transformed_distribution.density(limits=limits) for transformed_distribution in transformed_distributions]
# print(time.time()-T)
# b = tf.concat([tf.reshape(x,(*x.shape,1)) for x in a],axis=-1)
# b.shape
renormalization = np.sum(np.sum(image,axis=0),axis=0)[:,0].reshape(1,1,-1)
renormalization
def save_pic(data,name):
    renorm = np.sum(np.sum(data,axis=0),axis=0)[:].reshape(1,1,-1)/renormalization
    image_gen = np.clip(data/renorm,0,255).astype('uint8')
    im = Image.fromarray(image_gen)
    im.save(os.path.join(SAVE_FOLDER,'%s.png' % name))
# save_pic(b,'initial')
# b.shape



dataset = tf.data.Dataset.from_tensor_slices(dataset_as_tensor)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
dataset = dataset.shuffle(dataset_size)
dataset = dataset.batch(batch_size)
for transformed_distribution in transformed_distributions:
    transformed_distribution.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2)
    )



@tf.function
def aux(sample_batch,command_batch):
    return [transformed_distributions[i].reshape(transformed_distributions[i].score(sample_batch[:,i],command_batch[:,i])) for i in range(3)]

def train_step(weighted_sample_command_batch):
    weight_batch, sample_batch, command_batch = tf.split(
        weighted_sample_command_batch,[1,DISTRIBUTION_DIM,-1],
        axis=-1
    )
    weight_batch=tf.reshape(weight_batch,(-1,3))
    trainable_var = []
    for self in transformed_distributions:
        trainable_var += self.trainable_variables

    with tf.GradientTape() as tape:
        print(sample_batch.shape,command_batch.shape)
        i=0
        print(sample_batch[:,i].shape,command_batch[:,i].shape)
        scores = aux(sample_batch,command_batch)
        scores = [tf.reduce_mean(weight_batch[:,i]*scores[i]) for i in range(3)]
        loss = scores[0]+scores[1]+scores[2]
    grad = tape.gradient(loss, trainable_var)
    self.optimizer.apply_gradients(zip(grad, trainable_var))
for epoch in range(EPOCHS):
    i=0
    for batch in dataset:
        L = time.time()
        train_step(batch)
        i+=1
        print("epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-L))
    if epoch%10 == 0 and epoch>0:
        a = [transformed_distribution.density(limits=limits) for transformed_distribution in transformed_distributions]
        b = tf.concat([tf.reshape(x,(*x.shape,1)) for x in a],axis=-1)
        save_pic(b,'epoch_%03d' % epoch)

print("done in %s seconds" % (time.time()-T))
