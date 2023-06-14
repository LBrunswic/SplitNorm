import tensorflow as tf
import tensorflow_probability as tfp
from time import time
tfb = tfp.bijectors
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt

class CommandedTransformedDistribution(tf.keras.Model):
    def __init__(self, parametrized_transformed_distribution,_score,_prob,_sample, distribution_dim=2, command_dim=0, name='CommandedTransformedDistribution'):
        super(CommandedTransformedDistribution, self).__init__(name=name)
        self.parametrized_transformed_distribution = parametrized_transformed_distribution
        self._score = _score
        self._prob = _prob
        self._sample = _sample
        self.distribution_dim = distribution_dim
        self.command_dim = command_dim
        self.reshape = tf.keras.layers.Reshape((-1,))

    # @tf.function
    def score(self,sample_batch,command_batch):
        return self._score(sample_batch,command_batch)

    def prob(self,sample_batch,command_batch):
        return self._prob(sample_batch,command_batch)

    def sample(self,command_batch):
        return self._sample(command_batch)

    # @tf.function
    def _pre_compile(self, weighted_sample_command_batch):
            weight_batch, sample_batch, command_batch = weighted_sample_command_batch
            weight_batch=self.reshape(weight_batch)
            scores = self.reshape(self.score(sample_batch,command_batch))
            scores = weight_batch*scores
            loss = tf.reduce_mean(scores)
            return loss
    def train_step(self, weighted_sample_command_batch):
        # weight_batch, sample_batch, command_batch = tf.split(
        #     weighted_sample_command_batch,[1,self.distribution_dim,-1],
        #     axis=-1
        # )
        weight_batch, sample_batch, command_batch = weighted_sample_command_batch
        weight_batch=self.reshape(weight_batch)
        trainable_var = self.trainable_variables
        with tf.GradientTape() as tape:
            scores = self.reshape(self.score(sample_batch,command_batch))
            scores = weight_batch*scores
            loss = tf.reduce_mean(scores)
        grad = tape.gradient(loss, trainable_var)
        self.optimizer.apply_gradients(zip(grad, trainable_var))
        return {'loss':loss}

    def display_density(self,command=None,name='plop',limits=((-2,2,0.05),(-2,2,0.05))):
        if self.distribution_dim>2:
            raise NotImplementedError

        sample = np.mgrid[[slice(a,b,e) for a,b,e in limits]].transpose()
        x,y,_ = sample.shape
        sample = sample.reshape(-1,self.distribution_dim)
        if command is None:
            command = tf.ones((len(sample),self.command_dim))
        plt.matshow(
            self.prob(
                sample,
                command
            ).numpy().reshape(x,y)
        )
        plt.savefig(name)

    def density(self,command=None,name='plop',limits=((-2,2,0.05),(-2,2,0.05))):
        if self.distribution_dim>2:
            raise NotImplementedError

        sample = np.mgrid[[slice(a,b,e) for a,b,e in limits]].transpose()
        x,y,_ = sample.shape
        sample = sample.reshape(-1,self.distribution_dim)
        if command is None:
            command = tf.ones((len(sample),self.command_dim))
        return self.prob(
            sample,
            command
        ).numpy().reshape(x,y)


class FlowEnsemble(tf.keras.Model):
    def __init__(self, flow_family, command_dim=0,distribution_dim=2,name='Flowensemble'):
        super(FlowEnsemble, self).__init__(name=name)
        self.flow_family = flow_family
        self.reshape = tf.keras.layers.Reshape((1,))
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dot = tf.keras.layers.Dot(axes=-1)
        self.command_dim = command_dim
        self.distribution_dim = distribution_dim

    def build_dist(self,base_distribution,*build_args,**build_kwargs):
        transformed_distribution = tfd.TransformedDistribution(distribution=base_distribution, bijector=self.flow_family, *build_args, **build_kwargs)

        # @tf.function
        def _score(sample_batch,command_batch):
            return -transformed_distribution.log_prob(
                sample_batch,
                bijector_kwargs =  {'command': command_batch},
            )

        def _prob(sample_batch,command_batch,*args,**kwargs):
            return transformed_distribution.prob(
                sample_batch,
                *args,
                **{'bijector_kwargs': {'command': command_batch}},
                **kwargs
            )

        def _sample(command_batch,*args,**kwargs):
            sample_size = len(command_batch)
            weights = tf.ones((sample_size,1)) # (sample_size,1)
            samples = transformed_distribution.sample(
                sample_size,
                *args,
                **{'bijector_kwargs': {'command': command_batch}},
                **kwargs
            ) # (sample_size,sample_dim)
            return tf.concat([weights,samples],axis=1)

        return CommandedTransformedDistribution(
            transformed_distribution,
            _score,
            _prob,
            _sample,
            command_dim = self.command_dim,
            distribution_dim = self.distribution_dim,
        )


class ConvKernel(tf.keras.Model):
    def __init__(self, kernel_ensemble, channeller, commander, channel_dim=0, distribution_dim=2, command_dim=0,name='ConvFlow'):
        super(ConvKernel, self).__init__(name=name)
        self.kernel_ensemble = kernel_ensemble
        self.channeller = channeller
        self.commander = commander
        self.channel_dim = channel_dim
        self.command_dim = command_dim
        self.reshape = tf.keras.layers.Reshape((1,))
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dot = tf.keras.layers.Dot(axes=-1)
        self.distribution_dim = self.kernel_ensemble.distribution_dim
        self.channel_sample = channeller.output_shape[1]
        print(self.channel_sample)
    def build_dist(self,base_distribution):
        transformed_distribution = self.kernel_ensemble.build_dist(base_distribution)

        sample_batch = tf.keras.Input(shape=(self.distribution_dim,),name='dist_sample')
        command_batch = tf.keras.Input(shape=(self.command_dim,),name='command')
        channel_batch = tf.keras.Input(shape=(self.channel_dim,),name='channel')
        inputs = (sample_batch,channel_batch,command_batch)
        for x in (inputs):
            print(x.name,x.shape)
        outputs = (transformed_distribution.score(sample_batch,  self.commander((channel_batch, command_batch))))
        # outputs = tf.keras.layers.Reshape((1,))(outputs)
        # score_core = tf.keras.Model(inputs=inputs,outputs=ouputs)
        #
        # sample_batch = tf.keras.Input(shape=(self.distribution_dim,))
        # command_batch = tf.keras.Input(shape=(self.command_dim,))
        # channel_batchs = tf.keras.Input(shape=(self.channel_dim,))
        # inputs = (sample_batch,channel_batch,command_batch)
        # repeated_sample_batch = tf.keras.layers.RepeatVector(self.channel_sample)(sample_batch)
        # repeated_command_batch = tf.keras.layers.RepeatVector(self.channel_sample)(command_batch)
        # outputs = tf.keras.layers.TimeDistributed(score_core)((repeated_sample_batch,channel_batchs,repeated_command_batch))
        # channelled_score = tf.keras.Model(inputs=inputs,outputs=outputs)
        # print(channelled_score)
        dot = tf.keras.layers.Dot(axes=(0,1))
        # @tf.function
        def _score(sample_batch,command_batch):
            weighted_channel_batch = self.channeller((sample_batch, command_batch))
            # (data_batch_size,command_batch_size,command_dim+1)
            print('sample',sample_batch.shape)
            print('command',command_batch.shape)
            print('channel',weighted_channel_batch[:,0,:-1].shape)
            A = [
                self.reshape(transformed_distribution.score(sample_batch,  self.commander((channel_batch, command_batch))))
                for channel_batch in tf.unstack(weighted_channel_batch[:, :, :-1], axis=1)
            ]
            scores = self.concat(A)
            # A = channelled_score((sample_batch,weighted_channel_batch[:,:,-1],command_batch))
            weight_batch = weighted_channel_batch[:, :, -1]
            return sel.dot([scores, weight_batch])

        def _prob(sample_batch,command_batch,*args,**kwargs):
            weighted_channel_batch = self.channeller((sample_batch, command_batch))
            # (data_batch_size,command_batch_size,command_dim+1)
            A = [
                self.reshape(transformed_distribution.prob(sample_batch, self.commander((channel_batch, command_batch)),*args, **kwargs))
                for channel_batch in tf.unstack(weighted_channel_batch[:, :, :-1], axis=1)
            ]
            probs = self.concat(A)
            weight_batch = weighted_channel_batch[:, :, -1]
            return self.dot([probs, weight_batch])

        def _sample(channel_command_batch,*args,**kwargs):
            samples = transformed_distribution.sample(
                self.commander(tf.split(channel_command_batch,[self.channel_dim,self.command_dim],axis=-1)),
                *args,
                **kwargs
            )
            return samples

        return CommandedTransformedDistribution(
            transformed_distribution,
            _score,
            _prob,
            _sample,
            command_dim=self.command_dim,
            distribution_dim=self.distribution_dim
        )


class HigherConvKernel(tf.keras.Model):
    """ Convolutional Kernel taking a quantized distribution as input."""
    def __init__(self, kernel_ensemble=None, channeller=None, commander=None, quantization_dim=None,distribution_dim=2, channel_dim=0, command_dim=0,name='ConvFlow'):
        super(HigherConvKernel, self).__init__(name=name)
        self.kernel_ensemble = kernel_ensemble
        self.channeller = channeller
        self.commander = commander
        self.channel_dim = channel_dim
        self.command_dim = command_dim
        self.distribution_dim = distribution_dim
        self.quantization_dim = quantization_dim

    @tf.function
    def _aux(self,batch_quantized_dist_flat, command_batch_flat):
        return self.kernel.prob(batch_quantized_dist_flat,command_batch_flat)

    def reconstruction(self,batch_quantized_dist,command_batch):
        print(self)
        print(batch_quantized_dist.shape)
        print(command_batch.shape)
        reshape_command = tf.keras.layers.Reshape((1,-1))
        batch_size, quantization_dim, weight_distribution_dim = batch_quantized_dist.shape
        weighted_channels_batch = self.channeller(((batch_quantized_dist),command_batch)) # (data_batch_size,channel_batch_size,channel_dim+1)
        A = []
        for channel_batch in tf.unstack(weighted_channels_batch[:, :, :-1], axis=1):
            batch_quantized_dist_flat =  tf.reshape(batch_quantized_dist, (-1, batch_quantized_dist.shape[-1]))
            #(data_batch_size*quant_dim,dim+1)

            command_dim = channel_batch.shape[1]
            x = batch_quantized_dist[:,:,:1]*0 + reshape_command(self.commander((channel_batch,command_batch)))
            #    (batch_size,quant_dim,1)  +  (batch_size,1,command_dim) = (data_batch_size,quant_dim,command_dim)
            # TODO 1  : replace by a dynamic shape broadcast_to
            # https://stackoverflow.com/questions/57716363/explicit-broadcasting-of-variable-batch-size-tensor

            command_batch_flat = tf.reshape(x,(-1,command_dim))
            # (batch_size*quant_dim,command_dim)

            # print(batch_quantized_dist_flat[:,:,:-1].shape)
            # print(command_batch_flat.shape)
            # s = transformed_distribution.prob(batch_quantized_dist_flat[:,:,:-1],command_batch_flat)
            s = self._aux(batch_quantized_dist_flat[:,:-1], command_batch_flat)
            # (data_batch_size*quant_dim,dim+1),(batch_size*quant_dim,command_dim) -> (batch_size*quant_dim,)

            a = tf.reshape(s,(batch_size,quantization_dim,1))
            # (batch_size*quant_dim,)  ->  (batch_size,quant_dim,1)

            A.append(a)

        channel_densities = tf.concat(A,axis=-1)
        #[(batch_size,quant_dim,1) for _ in range(channel_batch_size)] -> (batch_size,quant_dim,channel_batch_size)

        channel_weights = tf.reshape(weighted_channels_batch[:, :, -1],(batch_size,1,-1))
        #(batch_size,channel_batch_size) -> (batch_size,1,channel_batch_size)

        densities = tf.reduce_sum(channel_densities*channel_weights,axis=-1)
        #(batch_size,quant_dim)
        return densities

    def build_dist(self,base_distribution):
        transformed_distribution = self.kernel_ensemble.build_dist(base_distribution)
        reshape = tf.keras.layers.Reshape((1,1))
        concat_prob = tf.keras.layers.Concatenate(axis=-2)
        reshape_total = tf.keras.layers.Reshape((self.quantization_dim,1))
        concat_total = tf.keras.layers.Concatenate(axis=-1)
        reshape_command = tf.keras.layers.Reshape((1,-1))
        flat = tf.keras.layers.Flatten()
        dot = tf.keras.layers.Dot(axes=-1)

        @tf.function
        def _aux(batch_quantized_dist_flat, command_batch_flat):
            return batch_quantized_dist_flat[:,-1]*tf.reshape(transformed_distribution.score(batch_quantized_dist_flat[:,:-1],command_batch_flat),(-1,))

        @tf.function
        def _score(batch_quantized_dist,command_batch):
            # A batch of distributions, command is given. Distribution are given
            # in a quantized way eg a batch of weighted samples of the distribution
            # is given. Therefore the input has shape
            #           ( batch_size, quantization_dim, distribution_dim + 1)
            # The Channeller take the whole as input  and return a batch of channels thus
            #  ( batch_size, channel_batch_size, channel_dim +1)
            # the channel_batch (without the weights) is split to given a list of
            # ( batch_size, channel_dim)
            # fed to the commander which provides
            # (batch_size, command_dim)
            #  we want to feed  (batch_size, quantization_dim, distribution_dim+1) efficiently
            # to the kernel below we thus reshape into (batch*quantization_dim,distribution_dim+1)
            # Also the command has to be broadcast_to (batch_size,quantization_dim,command_dim)
            # and reshaped the same way
            batch_size, quantization_dim, weight_distribution_dim = batch_quantized_dist.shape
            weighted_channels_batch = self.channeller(((batch_quantized_dist),command_batch)) # (data_batch_size,channel_batch_size,channel_dim+1)
            A = []
            for channel_batch in tf.unstack(weighted_channels_batch[:, :, :-1], axis=1):
                batch_quantized_dist_flat =  tf.reshape(batch_quantized_dist, (-1, batch_quantized_dist.shape[-1]))
                #(data_batch_size*quant_dim,dim+1)

                command_dim = channel_batch.shape[1]
                x = batch_quantized_dist[:,:,:1]*0 + reshape_command(self.commander((channel_batch,command_batch)))
                #    (batch_size,quant_dim,1)  +  (batch_size,1,command_dim) = (data_batch_size,quant_dim,command_dim)
                # TODO 1  : replace by a dynamic shape broadcast_to
                # https://stackoverflow.com/questions/57716363/explicit-broadcasting-of-variable-batch-size-tensor

                command_batch_flat = tf.reshape(x,(-1,command_dim))
                # (batch_size*quant_dim,command_dim)

                s = _aux(batch_quantized_dist_flat, command_batch_flat)
                # (data_batch_size*quant_dim,dim+1),(batch_size*quant_dim,command_dim) -> (batch_size*quant_dim,)

                a=tf.reduce_sum(tf.reshape(s,(-1,quantization_dim,1)),axis=1)
                # (batch_size*quant_dim,)  ->  (batch_size,quant_dim,1) -> (batch_size,1)

                A.append(a)

            scores = concat_total(A)  #(batch_size,command_dim)
            #[(batch_size,1) for _ in range(channel_batch_size)] -> (batch_size,channel_batch_size)

            weight_batch = weighted_channels_batch[:, :, -1]
            # (batch_size,channel_batch_size)

            score = dot([scores, weight_batch])
            # ()
            return score


        def _prob(sample_batch,command_batch,*args,**kwargs):
            weighted_channel_batch = self.channeller((sample_batch, command_batch))
            # (data_batch_size,command_batch_size,command_dim+1)
            A = [
                self.reshape(transformed_distribution.prob(sample_batch, self.commander((channel_batch, command_batch)),*args, **kwargs))
                for channel_batch in tf.unstack(weighted_channel_batch[:, :, :-1], axis=1)
            ]
            probs = self.concat(A)
            weight_batch = weighted_channel_batch[:, :, -1]
            return self.dot([probs, weight_batch])

        def _sample(channel_command_batch,*args,**kwargs):
            samples = transformed_distribution.sample(
                self.commander(tf.split(channel_command_batch,[self.channel_dim,self.command_dim],axis=-1)),
                *args,
                **kwargs
            )
            return samples

        out  = CommandedTransformedDistribution(
            transformed_distribution,
            _score,
            _prob,
            _sample,
            command_dim=self.command_dim,
            distribution_dim=self.distribution_dim
        )
        self.kernel = transformed_distribution
        self.transformed_distribution = out
        return out
