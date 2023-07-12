import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp;
import ConvolutionalKernel;
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skd
import time, os, datetime
import imageio.v3 as iio
from PIL import Image
import sys
# tf.config.optimizer.set_experimental_options({'disable_meta_optimizer':True})

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
EPOCHS = 200
STEP_PER_EPOCH = 30
DISTRIBUTION_DIM = 2
COMMAND_DIM = 0
SAVE_FOLDER = os.path.join('results',date_time_str)
os.makedirs(SAVE_FOLDER)
model_folder = os.path.join(SAVE_FOLDER,'model')
os.makedirs(model_folder)
ARCH = 0
LR = 1e-2
cutoff = 0
DATASET = 'MNIST'
add_x = True
TEST = False
# p = 0.5

if TEST:
    inward_depth = 1
    inward_width = 4
    switch_dim = 4
    batch_size = 32
    ensemble_size = 1
    # dataset_size = 28*28

    images = np.load('MNIST.npy')
    # QUANTIZATION_DIM = images[0].size
    QUANTIZATION_DIM = 8
else:
    flow_depth = 4
    flow_width = 64
    command_bypass = True
    time_bypass = True
    infra_command = 8

    channel_dim = infra_command
    channel_width = 64
    channel_depth = 4
    channel_sample = 1

    commander_depth = 1
    commander_width = 32

    # inward_depth = 4
    # inward_width = 32
    # switch_dim = inward_width
    # n_switch = 1
    # ensemble_size = 2


    batch_size = 32
    if DATASET == 'MNIST':
        images = np.load('MNIST.npy').reshape(-1,28,28)
    elif DATASET == 'CIFAR10':
        images = np.load('cifar_data.npy')
        images = images/np.sum(images,axis=1).reshape(-1,1)
        images = np.sum(images.reshape(-1,3,32,32),axis=1)

    # a=0
    # b=1
    # ensemble_size = 2
    # dataset_size = *28

    # images = np.load('MNIST.npy')[a:b+1]

    QUANTIZATION_DIM = images[0].size
    # QUANTIZATION_DIM = 64

alpha = 1.
delta = 1.
switch_arch = [
    [
        ConvolutionalKernel.CommandedConstructors.commanded_concat_sequential_dense_with_encoding,
        {
            'dim' : DISTRIBUTION_DIM,
            'cutoff' : cutoff,
            'add_x':add_x,
            'depth' : flow_depth,
            'width' : flow_width,
            'command_bypass':command_bypass,
            'time_bypass':time_bypass,
            'command_dim':infra_command,
            'kernelKWarg' : {'activation': lambda x: alpha*tf.tanh(delta*x)},
            # 'kernelKWarg' : {'activation':  tf.keras.layers.LeakyReLU()},
            # 'final_activation': lambda x: 0.3*tf.tanh(x),
        }
    ],
]

ffjord_core = ConvolutionalKernel.utils.build(switch_arch[ARCH])
# infra_command = ffjord_core.command_dim
solver_param ={
    'first_step_size' : 0.1,
    # 'max_num_steps': 1000,
    # 'atol':1e-2,
    # 'rtol':1e-3,
}
FFJORD_dorpri_arch1 = {
    'state_time_derivative_fn': ffjord_core,
    'ode_solve_fn': tfp.math.ode.DormandPrince(**solver_param).solve,
    'trace_augmentation_fn': tfp.bijectors.ffjord.trace_jacobian_exact
}
kernel_ensemble_arch = {
    'flow_family': tfp.bijectors.FFJORD(**FFJORD_dorpri_arch1),
    'command_dim': infra_command,
    'distribution_dim': DISTRIBUTION_DIM
}
kernel_ensemble = ConvolutionalKernel.FlowEnsemble(**kernel_ensemble_arch)

channeller_input_shape = (QUANTIZATION_DIM,DISTRIBUTION_DIM+1)
channeller_archs = [
    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
        {
            'distribution_shape': channeller_input_shape,
            'channel_dim': channel_dim,
            'command_dim': COMMAND_DIM,
            'channel_sample':channel_sample,
            'width':channel_width,
            'depth':channel_depth,
            'keep':...,
            # 'final_activation': lambda x: 2.*tf.tanh(x),
            # 'final_activation': lambda x: tf.nn.softmax(x,axis=2),
            # 'final_activation': tf.keras.layers.Activation('softmax'),
            # 'final_activation':lambda x: tf.clip_by_value(x,-3,3),
            'kernelKWarg' : {
                # 'activation': tf.tanh,
                'activation': tf.keras.layers.LeakyReLU(),
                # 'activation': tf.tanh,
                # 'kernel_initializer': tf.keras.initializers.GlorotNormal
            },
            # 'kernelKWarg' : {'activation': tf.keras.layers.Activation('tanh')},
            'final_rescale':1e-2,
            'weights_moderation' : lambda x: tf.tanh(x*1e-2)*3
        }
    ],
]

commander_arch1 = {
    'channel_dim' : channel_dim,
    'command_dim' : COMMAND_DIM,
    'output_dim' : infra_command,
}
commander = ConvolutionalKernel.CommanderConstructors.commander_passthrough(**commander_arch1)

# commander_arch1 = {
#     'channel_dim' : channel_dim,
#     'command_dim' : COMMAND_DIM,
#     'output_dim' : infra_command,
#      'depth': commander_depth,
#      'width': commander_width,
#      'kernelKWarg' : {'activation': tf.tanh},
#
# }
# commander = ConvolutionalKernel.CommanderConstructors.commander_sequential(**commander_arch1)


conv_kernel_arch = {
    'kernel_ensemble':kernel_ensemble,
    'commander': commander,
    'channeller': ConvolutionalKernel.utils.build(channeller_archs[ARCH]),
    'channel_dim': infra_command,
    'command_dim': COMMAND_DIM,
    'quantization_dim':QUANTIZATION_DIM
}
ConvKernel = ConvolutionalKernel.HigherConvKernel(**conv_kernel_arch)

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



def gen_sample_generator(pictures_coord,commands, quant_dim, delta_x, delta_y, max_batch_slice_size=2**11,field_coord=-1):
    noise_scale = np.array([[[delta_x,delta_y]]])
    rng = np.random.default_rng()
    indices = np.arange(pictures_coord.shape[0])
    weights = tf.ones((batch_size,1))
    while True:
        T = time.time()
        if batch_size!=pictures_coord.shape[0]:
            batch_indices = rng.choice(indices,size=batch_size)
        else:
            batch_indices = np.arange(pictures_coord.shape[0])
        picture_batch = pictures_coord[batch_indices]
        val = picture_batch[:,:, field_coord]
        val = val/np.sum(val,axis=1).reshape(-1,1)
        if quant_dim==picture_batch.shape[1]:
            chosen = picture_batch
        else:
            samples =  tf.random.categorical(tf.math.log(1e-8+val), quant_dim)
            chosen = tf.gather(picture_batch,samples,batch_dims=1)

        noise = tf.random.uniform([batch_size,quant_dim,DISTRIBUTION_DIM])*noise_scale
        coord = tf.concat([chosen[:,:,:DISTRIBUTION_DIM]+noise,chosen[:,:,field_coord:]],axis=-1)
        command = tf.gather(commands,batch_indices)
        res = (weights,coord,command)
        yield res


dataset_size = images.shape[0]
image = images[0]
image.shape
xmax, ymax = image.shape
commands = tf.ones((images.shape[0],transformed_distribution.command_dim))
delta_x,delta_y = 4/xmax, 4/ymax
limits = [(-2,2,delta_y),(-2,2,delta_x)]
picture_coord = np.mgrid[[slice(a, b, e) for a, b, e in limits]].transpose().astype('float32').reshape(1,image.size,2)
images_flat = images.reshape(dataset_size,image.size,1)
pictures_coord = np.concatenate([np.broadcast_to(picture_coord,(dataset_size,image.size,2)),images_flat],axis=-1)



test_indices = np.arange(batch_size)

output_signature = (
    tf.TensorSpec(shape=(batch_size,1),dtype=tf.float32,name='weights'),
    tf.TensorSpec(shape=(batch_size,QUANTIZATION_DIM,DISTRIBUTION_DIM+1),dtype=tf.float32,name='coordinates'),
    tf.TensorSpec(shape=(batch_size,COMMAND_DIM),dtype=tf.float32,name='command'),
)
dataset = tf.data.Dataset.from_generator(
    gen_sample_generator,
    output_signature=output_signature,
    args=(pictures_coord,commands,QUANTIZATION_DIM,delta_x,delta_y)
)

transformed_distribution.compile(
    optimizer=tf.keras.optimizers.Adam(LR)
)

def ttt(image):
    return (image/np.max(image)*255).numpy().astype('uint8')
for epoch in range(EPOCHS):
    i=0
    T = time.time()
    # transformed_distribution.fit(dataset,)
    for batch in dataset:

        # print(batch[1].shape)
        if i == STEP_PER_EPOCH:
            break
        L = time.time()
        transformed_distribution.train_step(batch)
        i+=1
        print("epoch %s, batch %s done in %s seconds        " % (epoch,i,time.time()-L))
    L = time.time()
    for batch in dataset:
        densities = ConvKernel.reconstruction(batch[1],batch[2])
        os.makedirs(os.path.join(SAVE_FOLDER,'epoch_%03d' % epoch))
        for i in range(batch_size):
            plt.imshow(tf.reshape(ttt(densities[i,:]),(xmax,ymax)))
            plt.savefig(os.path.join(SAVE_FOLDER,'epoch_%03d' % epoch,'%03d_a.png' % i))
            plt.close()
            # plt.clf()
            # plt.savefig(os.path.join(SAVE_FOLDER,'%03d_epoch_%03d_a.png' % (i,epoch)))
            plt.matshow(tf.reshape(batch[1][i,:,-1],(xmax,ymax)))
            plt.savefig(os.path.join(SAVE_FOLDER,'epoch_%03d' % epoch,'%03d_b.png' % i))
            # plt.clf()
            plt.close()
        break
    with open(os.path.join(SAVE_FOLDER,'channel_dist_%s' % epoch),'wb') as f:
        np.save(f,ConvKernel.channeller((pictures_coord,commands)))
    transformed_distribution.save_weights(os.path.join(model_folder,'weights_%s' % epoch))
    print('image generation:',time.time()-L)
    print("\n","epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-T))
    # transformed_distribution.
print("done in %s seconds" % (time.time()-T))
