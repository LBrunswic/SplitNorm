import tensorflow as tf;
import tensorflow_probability as tfp;
import ConvolutionalKernel;
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd
import time, os, datetime
import imageio.v3 as iio
from PIL import Image
import sys



GPU = 0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[GPU],'GPU')

tfd = tfp.distributions
now = datetime.datetime.now()
date_time_str = now.strftime("%m-%d_%Hh%Mm%Ss")


T = time.time()
EPOCHS = 10000
STEP_PER_EPOCH = 10
DISTRIBUTION_DIM = 2
COMMAND_DIM = 0
SAVE_FOLDER = os.path.join('results',date_time_str)
os.makedirs(SAVE_FOLDER)
model_folder = os.path.join(SAVE_FOLDER,'model')
os.makedirs(model_folder)
ARCH = 0
LR = 5*1e-3
cutoff = 0
add_x = True
TEST = False
p = 0.5

if TEST:
    inward_depth = 4
    inward_width = 16
    switch_dim = 16
    batch_size = 28*28
    dataset_size = 28*28
    image = tf.keras.datasets.mnist.load_data()[0][0][0]

else:
    inward_depth = 5
    inward_width = 32
    switch_dim = 32
    IMAGE = 'nasa_galaxy_xsmall.png'
    image = np.sum(iio.imread(os.path.join('images', IMAGE)),axis=-1)
    batch_size = 2**12
    dataset_size = image.size

switch_arch = [
    [
        ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense_with_encoding,
        {
            'dim' : DISTRIBUTION_DIM,
            'ensemble_size' : 1,
            'cutoff' : cutoff,
            'add_x':add_x,
            'n_switch' : 1,
            'inward_depth' : inward_depth,
            'inward_width' : inward_width,
            'switch_dim' : switch_dim,
            'kernelKWarg' : {'activation': tf.keras.activations.tanh},
        }
    ],
]

ffjord_core = ConvolutionalKernel.utils.build(switch_arch[ARCH])
infra_command = ffjord_core.command_dim

FFJORD_dorpri_arch1 = {
    'state_time_derivative_fn': ffjord_core,
    'ode_solve_fn': tfp.math.ode.DormandPrince().solve,
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
        ConvolutionalKernel.ChannellerConstructors.channeller_trivial,
        {
            'distribution_dim': DISTRIBUTION_DIM,
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
        }
    ],
]

conv_kernel_arch = {
    'kernel_ensemble':kernel_ensemble,
    'commander': ConvolutionalKernel.CommanderConstructors.commander_passthrough(**commander_arch1),
    'channeller': ConvolutionalKernel.utils.build(channeller_archs[ARCH]),
    'channel_dim': infra_command,
    'command_dim': COMMAND_DIM,
}
ConvKernel = ConvolutionalKernel.ConvKernel(**conv_kernel_arch)

# print model components
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


def gen_sample_generator(weighted_data, batch_size, delta_x, delta_y, max_batch_slice_size=2**11,p=p):
    val = weighted_data[:, 0]**p
    val = val/np.sum(val)
    noise_scale = np.array([[delta_x,delta_y]])
    batch_slice = [batch_size]
    if batch_size > max_batch_slice_size:
        batch_slice = [max_batch_slice_size for _ in range(batch_size//max_batch_slice_size)] + [ x for x in [batch_size%max_batch_slice_size] if x > 0]
    # print(batch_slice)

    def gen():
        while True:
            T = time.time()
            samples = tf.concat(
                [
                    tf.reshape(tf.random.categorical(tf.math.log(1e-8+val.reshape(1,-1)), batch_slice_size), (-1,))
                    for batch_slice_size in batch_slice
                ],
                axis=0
            )
            noise = tf.random.uniform([batch_size,DISTRIBUTION_DIM])*noise_scale
            # print('samples',samples.shape)
            # print('data:',weighted_data.shape)
            # print('noise',noise.shape)
            res = tf.concat(
                [
                    weighted_data[samples][:,:1]**(1-p),
                    weighted_data[samples][:,1:1+DISTRIBUTION_DIM]+noise,
                    weighted_data[samples][:,1+DISTRIBUTION_DIM:]
                ],
                axis=1
            )
            print('batch generation time:',time.time()-T)
            yield res
    return gen()

xmax, ymax = image.shape
commands = tf.ones((*image.shape,transformed_distribution.command_dim))
image = image.reshape(*image.shape,1)
delta_x,delta_y = 4/xmax, 4/ymax
limits = [(-2,2,delta_y),(-2,2,delta_x)]
sample = np.mgrid[[slice(a, b, e) for a, b, e in limits]].transpose().astype('float32')
dataset_as_tensor = np.concatenate([image,sample,commands],axis=-1).reshape(xmax*ymax,-1)
plt.matshow(tf.reshape(dataset_as_tensor[:,0],(xmax,ymax)))
plt.savefig(os.path.join(SAVE_FOLDER,'target.png'))


dataset = tf.data.Dataset.from_generator(
    gen_sample_generator,
    output_signature=tf.TensorSpec(shape=(batch_size,3),dtype=tf.float32),
    args=(dataset_as_tensor,batch_size,delta_x,delta_y)
)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
# dataset = dataset.shuffle(dataset_size)
# dataset = dataset.batch(batch_size)
transformed_distribution.compile(
    optimizer=tf.keras.optimizers.Adam(LR)
)

for epoch in range(EPOCHS):
    i=0
    T = time.time()
    for batch in dataset:
        if i > STEP_PER_EPOCH:
            break
        L = time.time()
        transformed_distribution.train_step(batch)
        i+=1
        print("epoch %s, batch %s done in %s seconds        " % (epoch,i,time.time()-L))
    transformed_distribution.display_density(
        name=os.path.join(SAVE_FOLDER,'epoch_%03d_number_%02d.png' % (0,epoch)),
        limits=limits
    )
    print("\n","epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-T))

print("done in %s seconds" % (time.time()-T))
