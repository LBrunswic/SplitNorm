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
# tf.config.optimizer.set_experimental_options({'disable_meta_optimizer':True})


GPU = 0
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[GPU],'GPU')

tfd = tfp.distributions
now = datetime.datetime.now()
date_time_str = now.strftime("%m-%d_%Hh%Mm%Ss")


T = time.time()
EPOCHS = 10000
STEP_PER_EPOCH = 100
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
# p = 0.5

if TEST:
    inward_depth = 1
    inward_width = 4
    switch_dim = 4
    batch_size = 32
    ensemble_size = 1
    # dataset_size = 28*28
    images = np.load('MNIST.npy')
    QUANTIZATION_DIM = images[0].size
else:
    inward_depth = 3
    inward_width = 8
    switch_dim = 8
    batch_size = 8
    ensemble_size = 3
    # dataset_size = 28*28
    images = np.load('MNIST.npy')
    QUANTIZATION_DIM = images[0].size

switch_arch = [
    [
        ConvolutionalKernel.CommandedConstructors.commanded_switched_ensemble_sequential_dense_with_encoding,
        {
            'dim' : DISTRIBUTION_DIM,
            'ensemble_size' : ensemble_size,
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

channeller_input_shape = (QUANTIZATION_DIM,DISTRIBUTION_DIM+1)
channeller_archs = [
    [
        ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
        {
            'distribution_shape': channeller_input_shape,
            'channel_dim': infra_command,
            'command_dim': COMMAND_DIM,
            'channel_sample':3,
            'width':16,
            'depth':2,
            'keep':(...,-1)
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

# #
# def gen_sample_generator(dataset_as_tensor,commands, batch_size, delta_x, delta_y, max_batch_slice_size=2**11,p=p):
#     noise_scale = np.array([[delta_x,delta_y]])
#     batch_slice = [batch_size]
#     if batch_size > max_batch_slice_size:
#         batch_slice = [max_batch_slice_size for _ in range(batch_size//max_batch_slice_size)] + [ x for x in [batch_size%max_batch_slice_size] if x > 0]
#
#     def gen():
#         while True:
#             T = time.time()
#             samples = tf.concat(
#                 [
#                     tf.reshape(tf.random.categorical(tf.math.log(1e-8+val.reshape(1,-1)), batch_slice_size), (-1,))
#                     for batch_slice_size in batch_slice
#                 ],
#                 axis=0
#             )
#             noise = tf.random.uniform([batch_size,DISTRIBUTION_DIM])*noise_scale
#             res = tf.concat(
#                 [
#                     weighted_data[samples][:,:1]**(1-p),
#                     weighted_data[samples][:,1:1+DISTRIBUTION_DIM]+noise,
#                     weighted_data[samples][:,1+DISTRIBUTION_DIM:]
#                 ],
#                 axis=1
#             )
#             print('batch generation time:',time.time()-T)
#             yield res
#     return gen()
np.broadcast_to
image = images[0]
dataset_size = images.shape[0]
image.shape
xmax, ymax = image.shape
commands = tf.ones((images.shape[0],transformed_distribution.command_dim))
delta_x,delta_y = 4/xmax, 4/ymax
limits = [(-2,2,delta_y),(-2,2,delta_x)]
picture_coord = np.mgrid[[slice(a, b, e) for a, b, e in limits]].transpose().astype('float32').reshape(1,image.size,2)
images_flat = images.reshape(dataset_size,image.size,1)
dataset_as_tensor = np.concatenate([np.broadcast_to(picture_coord,(dataset_size,image.size,2)),images_flat],axis=-1)
dataset_as_tensor.shape
commands.shape
for i in range(3):
    plt.matshow(tf.reshape(dataset_as_tensor[i,:,-1],(xmax,ymax)))
    plt.savefig(os.path.join(SAVE_FOLDER,'target_%s.png' % i))
    plt.clf()
#
# dataset = tf.data.Dataset.from_generator(
#     gen_sample_generator,
#     output_signature=tf.TensorSpec(shape=(batch_size,3),dtype=tf.float32),
#     args=(dataset_as_tensor,batch_size,delta_x,delta_y)
# )
dataset = tf.data.Dataset.from_tensor_slices(dataset_as_tensor)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
dataset = dataset.cache()
transformed_distribution.compile(
    optimizer=tf.keras.optimizers.Adam(LR)
)
a = dataset.as_numpy_iterator()
next(a).shape
for epoch in range(EPOCHS):
    i=0
    T = time.time()
    dataset.shuffle(60000)
    for batch in dataset:
        print(batch.shape)
        if i > STEP_PER_EPOCH:
            break
        L = time.time()
        transformed_distribution.train_step((tf.ones((batch_size,),dtype=tf.float32),batch,tf.zeros((batch_size,0),dtype=tf.float32)))
        i+=1
        print("epoch %s, batch %s done in %s seconds        " % (epoch,i,time.time()-L))
    densities = ConvKernel.reconstruction(dataset_as_tensor[:3],tf.zeros((3,0)))
    for i in range(3):
        plt.matshow(tf.reshape(densities[i,:],(xmax,ymax)))
        plt.savefig(os.path.join(SAVE_FOLDER,'target_%03d_epoch_%03d.png' % (i,epoch)))
        plt.clf()

    print("\n","epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-T))

print("done in %s seconds" % (time.time()-T))
