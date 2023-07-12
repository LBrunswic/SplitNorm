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
tfd = tfp.distributions
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


def train(
    EPOCHS = 200,
    STEP_PER_EPOCH = 30,
    DISTRIBUTION_DIM = 2,
    COMMAND_DIM = 0,
    ARCH = 0,
    LR = 1e-2,
    cutoff = 0,
    add_x = True,
    flow_depth = 4,
    flow_width = 64,
    channel2_dim = 8,
    channel1_width = 32,
    channel1_depth = 2,
    channel1_dim = 8,
    command1_bypass = True,
    time1_bypass = True,

    channel2_width = 64,
    channel2_depth = 4,
    channel2_sample = 1,
    command2_bypass = True,
    time2_bypass = True,

    batch_size = 32,
    alpha = 1.,
    delta = 1.,
    DATASET = 'MNIST',
):

    now = datetime.datetime.now()
    date_time_str = now.strftime("%m-%d_%Hh%Mm%Ss")
    T = time.time()
    lvl1_command = channel2_dim
    channel1_sample = channel1_dim

    DATASETS = {
        'MNIST' : {
            'file' : 'MNIST.npy',
            'shape' : (-1,28,28)
        },
        'FASHION' : {
            'file' : 'fashion.npy',
            'shape' : (-1,28,28)
        },

    }
    images = np.load(DATASETS[DATASET]['file']).reshape(DATASETS[DATASET]['shape'])
    SAVE_FOLDER = os.path.join('results',date_time_str)
    os.makedirs(SAVE_FOLDER)
    model_folder = os.path.join(SAVE_FOLDER,'model')
    os.makedirs(model_folder)
    QUANTIZATION_DIM = images[0].size
    print('lvl1_command',lvl1_command)
# Level 1 ARCH
    flow_arch = [
        [
            ConvolutionalKernel.CommandedConstructors.commanded_concat_sequential_dense_with_encoding,
            {
                'dim' : DISTRIBUTION_DIM,
                'cutoff' : cutoff,
                'add_x':add_x,
                'depth' : flow_depth,
                'width' : flow_width,
                'command_bypass':command1_bypass,
                'time_bypass':time1_bypass,
                'command_dim':channel1_dim+lvl1_command,
                'kernelKWarg' : {'activation': lambda x: alpha*tf.tanh(delta*x)},
            }
        ],
    ]
    ffjord_core = ConvolutionalKernel.utils.build(flow_arch[ARCH])

    solver_param ={
        'first_step_size' : 0.1,
    }
    FFJORD_dorpri_arch1 = {
        'state_time_derivative_fn': ffjord_core,
        'ode_solve_fn': tfp.math.ode.DormandPrince(**solver_param).solve,
        'trace_augmentation_fn': tfp.bijectors.ffjord.trace_jacobian_exact
    }
    kernel_ensemble1_arch = {
        'flow_family': tfp.bijectors.FFJORD(**FFJORD_dorpri_arch1),
        'command_dim': lvl1_command+channel1_dim,
        'distribution_dim': DISTRIBUTION_DIM
    }
    kernel_ensemble1 = ConvolutionalKernel.FlowEnsemble(**kernel_ensemble1_arch)

    commander1_arch = [
        [
            ConvolutionalKernel.CommanderConstructors.commander_passthrough,
            {
                'channel_dim' : channel1_dim,
                'command_dim' : lvl1_command,
                'output_dim' : channel1_dim+lvl1_command,
                'name': 'lvl1_commander'
            }
        ],
    ]

    channeller1_input_shape = (DISTRIBUTION_DIM,)
    channeller1_archs = [
        [
            ConvolutionalKernel.ChannellerConstructors.channeller_sequential_finite,
            {
                'distribution_dim': DISTRIBUTION_DIM,
                'channel_dim': channel1_dim,
                'command_dim': lvl1_command,
                'width':32,
                'depth':3,
                'finite_set':ConvolutionalKernel.utils.switch_commands(channel1_dim,1)
            }
        ],
    ]

    lvl1_kernel_arch = {
        'kernel_ensemble': kernel_ensemble1,
        'commander': ConvolutionalKernel.utils.build(commander1_arch[ARCH]),
        'channeller': ConvolutionalKernel.utils.build(channeller1_archs[ARCH]),
        'channel_dim': channel1_dim,
        'command_dim': lvl1_command,
    }
    lvl1_kernel = ConvolutionalKernel.ConvKernel(**lvl1_kernel_arch)

# LEVEL 2 arch

    channeller2_input_shape = (QUANTIZATION_DIM,DISTRIBUTION_DIM+1)
    channeller2_archs = [
        [
            ConvolutionalKernel.ChannellerConstructors.channeller_sequential,
            {
                'distribution_shape': channeller2_input_shape,
                'channel_dim': channel2_dim,
                'command_dim': 0,
                'channel_sample':channel2_sample,
                'width':channel2_width,
                'depth':channel2_depth,
                'keep':...,
                'kernelKWarg' : {
                    'activation': tf.keras.layers.LeakyReLU(),
                },
                'name':'lvl2_channeller',
                'final_rescale':1e-3,
                'weights_moderation' : lambda x: tf.tanh(x*1e-2)*3
            }
        ],
    ]

    commander2_arch = {
        'channel_dim' : channel2_dim,
        'command_dim' : 0,
        'output_dim' : lvl1_command,
        'name':'lvl2_commander',
    }
    commander2 = ConvolutionalKernel.CommanderConstructors.commander_passthrough(**commander2_arch)
    conv_kernel2_arch = {
        'kernel_ensemble':lvl1_kernel,
        'commander': commander2,
        'channeller': ConvolutionalKernel.utils.build(channeller2_archs[ARCH]),
        'channel_dim': channel2_dim,
        'command_dim': 0,
        'quantization_dim':QUANTIZATION_DIM
    }
    ConvKernel = ConvolutionalKernel.HigherConvKernel(**conv_kernel2_arch)
    models = [
        ffjord_core.kernel,
        lvl1_kernel.channeller,
        lvl1_kernel.commander,
        ConvKernel.channeller,
        ConvKernel.commander
    ]
    for model in models:
        print(model)
        print(model.name)
        print("________________")
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
        for batch in dataset:
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
                plt.matshow(tf.reshape(batch[1][i,:,-1],(xmax,ymax)))
                plt.savefig(os.path.join(SAVE_FOLDER,'epoch_%03d' % epoch,'%03d_b.png' % i))
                plt.close()
            break
        with open(os.path.join(SAVE_FOLDER,'channel_dist_%s' % epoch),'wb') as f:
            np.save(f,ConvKernel.channeller((pictures_coord,commands)))
        transformed_distribution.save_weights(os.path.join(model_folder,'weights_%s' % epoch))
        print('image generation:',time.time()-L)
        print("\n","epoch %s, batch %s done in %s seconds" % (epoch,i,time.time()-T))
    print("done in %s seconds" % (time.time()-T))

from TrainingHP import HPset
for HP in HPset:
    tf.keras.backend.clear_session()
    train(**HP)
