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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    GPU = int(sys.argv[1])
except:
    GPU = -1
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    if GPU<0:
        tf.config.set_visible_devices([],'GPU')
    else:
        tf.config.set_visible_devices(gpus[GPU],'GPU')

tfd = tfp.distributions
now = datetime.datetime.now()
date_time_str = now.strftime("%m-%d_%Hh%Mm%Ss")


T = time.time()
EPOCHS = 100
STEP_PER_EPOCH = 50
DISTRIBUTION_DIM = 2
COMMAND_DIM = 0
SAVE_FOLDER = os.path.join('results',date_time_str)
os.makedirs(SAVE_FOLDER)
model_folder = os.path.join(SAVE_FOLDER,'model')
os.makedirs(model_folder)
ARCH = 1
LR = 1e-2
cutoff = 0
add_x = True
TEST = True
p = 0.5
IMAGE = 'nasa_galaxy_xsmall.png'

if TEST:
    inward_depth = 4
    inward_width = 64
    switch_dim = 64

    i = 12
    image = np.load('cifar_data.npy')[i].reshape(3,32,32).transpose().reshape(32,32*3).astype('float32')/255
    x_size,y_size, picture_channel = 32,32,3
    batch_size = image.size
    dataset_size = image.size

    channel_dim = 5
    infra_command = channel_dim

else:
    inward_depth = 4
    inward_width = 64
    switch_dim = 64
    image = iio.imread(os.path.join('images', IMAGE)).astype('float32')
    x_size,y_size, picture_channel = image.shape
    image = image.reshape(x_size,y_size*picture_channel)
    channel_dim = 5
    infra_command = channel_dim
    # image = tf.keras.datasets.mnist.load_data()[0][0][1]
    # batch_size = 2**10
    batch_size = 2**13
    dataset_size = image.size

switch_arch = [
    [
        ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense_with_encoding,
        {
            'dim' : DISTRIBUTION_DIM,
            'ensemble_size' : 3,
            'cutoff' : cutoff,
            'add_x':add_x,
            'n_switch' : 1,
            'inward_depth' : inward_depth,
            'inward_width' : inward_width,
            'switch_dim' : switch_dim,
            'kernelKWarg' : {'activation': tf.keras.activations.tanh},
        }
    ],
    [
    ConvolutionalKernel.CommandedConstructors.commanded_concat_sequential_dense_with_encoding,
    {
        'dim' : DISTRIBUTION_DIM,
        'cutoff' : cutoff,
        'add_x':add_x,
        'command_dim' : infra_command,
        'depth' : inward_depth,
        'width' : inward_width,
        'kernelKWarg' : {'activation': tf.keras.activations.tanh},
    }
    ],

]

ffjord_core = ConvolutionalKernel.utils.build(switch_arch[ARCH])
infra_command = ffjord_core.command_dim

solver_param ={
    'first_step_size' : 0.1,
    'max_num_steps': None
}


FFJORD_dorpri_arch1 = {
    'state_time_derivative_fn': ffjord_core,
    'ode_solve_fn': tfp.math.ode.DormandPrince(**solver_param).solve,
    # 'ode_solve_fn': ode_solve_fn,
    'trace_augmentation_fn': tfp.bijectors.ffjord.trace_jacobian_exact
}
kernel_ensemble_arch = {
    'flow_family': tfp.bijectors.FFJORD(**FFJORD_dorpri_arch1),
    'command_dim': infra_command,
    'distribution_dim': DISTRIBUTION_DIM
}
kernel_ensemble = ConvolutionalKernel.FlowEnsemble(**kernel_ensemble_arch)

commander_arch = [
    [
        ConvolutionalKernel.CommanderConstructors.commander_passthrough,
        {
            'channel_dim' : channel_dim,
            'command_dim' : COMMAND_DIM,
            'output_dim' : infra_command,
        }
    ],
    [
        ConvolutionalKernel.CommanderConstructors.commander_passthrough,
        {
            'channel_dim' : channel_dim,
            'command_dim' : COMMAND_DIM,
            'output_dim' : infra_command,
        }
    ],
]

channeller_archs = [
    # [
    #     ConvolutionalKernel.ChannellerConstructors.channeller_trivial,
    #     {
    #         'distribution_dim': DISTRIBUTION_DIM,
    #         'channel_dim': infra_command,
    #         'command_dim': COMMAND_DIM,
    #     }
    # ],

    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential_finite,
        {
            'distribution_dim': DISTRIBUTION_DIM,
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
            'width':32,
            'depth':4,
            'finite_set':ConvolutionalKernel.utils.switch_commands(switch_arch[0][1]['ensemble_size'],switch_arch[0][1]['n_switch'])
        }
    ],
    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
        {
            'distribution_dim': DISTRIBUTION_DIM,
            'channel_dim': channel_dim,
            'command_dim': COMMAND_DIM,
            'channel_sample':3,
            'width':64,
            'depth':4,
        }
    ],
]

conv_kernel_arch = {
    'kernel_ensemble':kernel_ensemble,
    'commander': ConvolutionalKernel.utils.build(commander_arch[ARCH]),
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
    indices = tf.constant(np.arange(batch_size))
    logit = weighted_data[:, 0]**p
    all_weights = weighted_data[:, :1]**(1-p)
    logit = tf.constant(logit.reshape(1,-1)/np.sum(logit))
    logit = tf.math.log(1e-8+logit)
    noise_scale = np.array([[delta_x,delta_y]])
    batch_slice = [batch_size]
    data = tf.constant(weighted_data)
    if batch_size > max_batch_slice_size:
        batch_slice = [max_batch_slice_size for _ in range(batch_size//max_batch_slice_size)] + [ x for x in [batch_size%max_batch_slice_size] if x > 0]
    while True:
        assert(np.max(np.abs(data-weighted_data))<1e-5)
        print('________________')
        T = time.time()
        if dataset_size != batch_size:
            samples = tf.concat(
                [
                    tf.reshape(tf.random.categorical(logit, batch_slice_size), (batch_slice_size,))
                    for batch_slice_size in batch_slice
                ],
                axis=0
            )
        else:
            samples = indices
        samples = tf.reshape(samples,(batch_size,))
        print('samples',samples.shape)
        noise = tf.random.uniform([batch_size,DISTRIBUTION_DIM])*noise_scale
        chosen = tf.gather(data,samples)
        print('chosen',chosen.shape)
        weights = tf.gather(all_weights,samples)
        coord = chosen[:,1:1+DISTRIBUTION_DIM]+noise
        command = chosen[:,1+DISTRIBUTION_DIM:]
        res = (weights, coord,  command)
        print('batch generation time:',time.time()-T)
        print('weights',weights.shape)
        print('coord',coord.shape)
        print('command',command.shape)
        print('________________')
        yield res


xmax, ymax = image.shape
commands = tf.ones((*image.shape,transformed_distribution.command_dim))
image = image.reshape(*image.shape,1)
delta_x,delta_y = 4/xmax, 4/ymax
limits = [(-2,2,delta_y),(-2,2,delta_x)]
sample = np.mgrid[[slice(a, b, e) for a, b, e in limits]].transpose().astype('float32')
dataset_as_tensor = np.concatenate([image,sample,commands],axis=-1).reshape(xmax*ymax,-1)
plt.imshow(tf.reshape(dataset_as_tensor[:,0],(x_size,y_size,picture_channel)))
plt.savefig(os.path.join(SAVE_FOLDER,'target.png'))

output_signature = (
    tf.TensorSpec(shape=(batch_size,1),dtype=tf.float32,name='weights'),
    tf.TensorSpec(shape=(batch_size,DISTRIBUTION_DIM),dtype=tf.float32,name='coordinates'),
    tf.TensorSpec(shape=(batch_size,COMMAND_DIM),dtype=tf.float32,name='command'),
)
dataset = tf.data.Dataset.from_generator(
    gen_sample_generator,
    output_signature=output_signature,
    args=(dataset_as_tensor,batch_size,delta_x,delta_y)
)


transformed_distribution.compile(
    optimizer=tf.keras.optimizers.Adam(LR)
)
limits = [(-2,2,delta_y),(-2,2,delta_x)]

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
    P = time.time()
    transformed_distribution.display_picture(
        name=os.path.join(SAVE_FOLDER,'epoch_%03d_number_%02d.png' % (0,epoch)),
        limits=limits
    )
    print('image generation :',time.time ()-P)
    print("\n","epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-T))

print("done in %s seconds" % (time.time()-T))
