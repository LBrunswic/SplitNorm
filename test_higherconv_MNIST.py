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
EPOCHS = 1
STEP_PER_EPOCH = 1
DISTRIBUTION_DIM = 2
COMMAND_DIM = 0
SAVE_FOLDER = os.path.join('results',date_time_str)
os.makedirs(SAVE_FOLDER)
model_folder = os.path.join(SAVE_FOLDER,'model')
os.makedirs(model_folder)
ARCH = 0
LR = 1e-2
cutoff = 0
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
    inward_depth = 4
    inward_width = 64
    switch_dim = inward_width
    n_switch = 1
    ensemble_size = 5


    # inward_depth = 4
    # inward_width = 32
    # switch_dim = inward_width
    # n_switch = 1
    # ensemble_size = 2


    batch_size = 32
    # ensemble_size = 2
    # dataset_size = 28*28
    images = np.load('MNIST.npy')
    QUANTIZATION_DIM = images[0].size
    # QUANTIZATION_DIM = 64


switch_arch = [
    [
        ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense_with_encoding,
        {
            'dim' : DISTRIBUTION_DIM,
            'ensemble_size' : ensemble_size,
            'cutoff' : cutoff,
            'add_x':add_x,
            'n_switch' : n_switch,
            'inward_depth' : inward_depth,
            'inward_width' : inward_width,
            'switch_dim' : switch_dim,
            'kernelKWarg' : {'activation': tf.keras.activations.tanh},
        }
    ],
]

ffjord_core = ConvolutionalKernel.utils.build(switch_arch[ARCH])
infra_command = ffjord_core.command_dim

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
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
            'channel_sample':3,
            'width':32,
            'depth':3,
            'keep':(...,-1),
            'final_activation':tf.keras.activations.tanh
        }
    ],
]

commander_arch1 = {
    'channel_dim' : infra_command,
    'command_dim' : COMMAND_DIM,
    'output_dim' : infra_command,
}
commander = ConvolutionalKernel.CommanderConstructors.commander_passthrough(**commander_arch1)
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



def gen_sample_generator(pictures_coord,commands, batch_size, quant_dim, delta_x, delta_y, max_batch_slice_size=2**11,field_coord=-1):
    noise_scale = np.array([[[delta_x,delta_y]]])
    rng = np.random.default_rng()
    indices = np.arange(pictures_coord.shape[0])
    weights = tf.ones((batch_size,1))

    while True:
        T = time.time()
        batch_indices = rng.choice(indices,size=batch_size)
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
        print(weights.shape)
        print(coord.shape)
        print(command.shape)
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



test_indices = np.array([1,3,5,7,2,0,13,15,17,22])
# print(pictures_coord[test_indices].shape)
# densities = ConvKernel.reconstruction(pictures_coord[test_indices],tf.zeros((test_indices.size,0)))
# for i in range(10):
#     plt.matshow(tf.reshape(densities[i,:],(xmax,ymax)))
#     plt.savefig(os.path.join(SAVE_FOLDER,'target_%03d_epoch_%03d.png' % (i,epoch)))
#     plt.clf()
# raise
for i in range(10):
    plt.matshow(tf.reshape(pictures_coord[test_indices][i,:,-1],(xmax,ymax)))
    os.makedirs(os.path.join(SAVE_FOLDER,'example_%s' %i))
    plt.savefig(os.path.join(SAVE_FOLDER,'example_%s' %i, 'target.png'))
    plt.clf()

output_signature = (
    tf.TensorSpec(shape=(batch_size,1),dtype=tf.float32,name='weights'),
    tf.TensorSpec(shape=(batch_size,QUANTIZATION_DIM,DISTRIBUTION_DIM+1),dtype=tf.float32,name='coordinates'),
    tf.TensorSpec(shape=(batch_size,COMMAND_DIM),dtype=tf.float32,name='command'),
)
dataset = tf.data.Dataset.from_generator(
    gen_sample_generator,
    output_signature=output_signature,
    args=(pictures_coord,commands,batch_size,QUANTIZATION_DIM,delta_x,delta_y)
)

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
    L = time.time()
    densities = ConvKernel.reconstruction(pictures_coord[test_indices],tf.zeros((test_indices.size,0)))
    for i in range(10):
        plt.matshow(tf.reshape(densities[i,:],(xmax,ymax)))
        plt.savefig(os.path.join(SAVE_FOLDER,'example_%s' %i,'epoch_%03d.png' % epoch))
        plt.clf()
    with open(os.path.join(SAVE_FOLDER,'channel_dist_%s' % epoch),'wb') as f:
        np.save(f,ConvKernel.channeller((pictures_coord,commands)))
    transformed_distribution.save_weights(os.path.join(model_folder,'weights_%s' % epoch))
    print('image generation:',time.time()-L)
    print("\n","epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-T))
    # transformed_distribution.
print("done in %s seconds" % (time.time()-T))
