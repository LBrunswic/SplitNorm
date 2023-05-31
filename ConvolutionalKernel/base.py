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

    @tf.function
    def score(self,sample_batch,command_batch):
        return self._score(sample_batch,command_batch)

    def prob(self,sample_batch,command_batch):
        return self._prob(sample_batch,command_batch)

    def sample(self,command_batch):
        return self._sample(command_batch)

    def train_step(self, weighted_sample_command_batch):
        weight_batch, sample_batch, command_batch = tf.split(
            weighted_sample_command_batch,[1,self.distribution_dim,-1],
            axis=-1
        )
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

        @tf.function
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
    def __init__(self, kernel_ensemble, channeller, commander, channel_dim=0,distribution_dim=2, command_dim=0,name='ConvFlow'):
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

    def build_dist(self,base_distribution):
        transformed_distribution = self.kernel_ensemble.build_dist(base_distribution)
        @tf.function
        def _score(sample_batch,command_batch):
            weighted_channel_batch = self.channeller((sample_batch, command_batch))
            # (data_batch_size,command_batch_size,command_dim+1)
            A = [
                self.reshape(transformed_distribution.score(sample_batch,  self.commander((channel_batch, command_batch))))
                for channel_batch in tf.unstack(weighted_channel_batch[:, :, :-1], axis=1)
            ]
            scores = self.concat(A)
            weight_batch = weighted_channel_batch[:, :, -1]
            return self.dot([scores, weight_batch])

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
def __init__(self, kernel_ensemble, channeller, commander, quantization_dim=None,distribution_dim=2, channel_dim=0, command_dim=0,name='ConvFlow'):
    super(HigherConvKernel, self).__init__(name=name)
    self.kernel_ensemble = kernel_ensemble
    self.channeller = channeller
    self.commander = commander
    self.channel_dim = channel_dim
    self.command_dim = command_dim
    self.distribution_dim = distribution_dim

def build_dist(self,base_distribution):
    transformed_distribution = self.kernel_ensemble.build_dist(base_distribution)
    @tf.function
    def _score(sample_batch,command_batch):
        weighted_channel_batch = self.channeller((sample_batch, command_batch))
        # (data_batch_size,command_batch_size,infra_command_dim+1)


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

# class ConvFlowV2(tf.keras.Model):
#     """Multi-layer NN ode_fn."""
#     def __init__(self, flow, channel,quantization_dim, channel_kernel_dim,name='ConvFlow',
#                  base_distribution_gen = tfd.MultivariateNormalDiag,
#                  base_distributionKWarg = None
#                  ):
#         super(ConvFlowV2, self).__init__(name=name)
#         self.flow = flow
#         self.channel = channel
#         self.quantization_dim = quantization_dim
#         self.channel_kernel_dim = channel_kernel_dim
#         self.base_distribution_gen = base_distribution_gen
#         self.base_distributionKWarg = base_distributionKWarg
#         self.reshape = tf.keras.layers.Reshape((1,1))
#         self.concat_prob = tf.keras.layers.Concatenate(axis=-2)
#         self.reshape_total = tf.keras.layers.Reshape((self.quantization_dim,1))
#         self.concat_total = tf.keras.layers.Concatenate(axis=-1)
#         self.reshape_command = tf.keras.layers.Reshape((1,-1))
#         self.dot = tf.keras.layers.Dot(axes=-1)
#         self.base_distribution = self.base_distribution_gen(**self.base_distributionKWarg)
#         self.transformed_distribution = tfd.TransformedDistribution(distribution=self.base_distribution, bijector=self.flow)
#
#     @tf.function
#     def _aux(self,quanta,command_batch):
#         return self.reshape(quanta[:, :1]) * self.reshape(-self.transformed_distribution.log_prob(
#             quanta[:, 1:],
#             **{'bijector_kwargs': {'command': command_batch}}
#         ))
#
#     def pre_score(self, batch_quantized_dist):  # (batch_size,quantization_dim,dim+1)
#         batch_quantized_dist_no_coord = batch_quantized_dist[:, :, :self.channel_kernel_dim]
#         weighted_command_batch = self.channel(
#             batch_quantized_dist_no_coord)  # (data_batch_size,command_batch_size,command_dim+1)
#         for command_batch in tf.unstack(weighted_command_batch, axis=1):
#             for quanta in tf.unstack(batch_quantized_dist, axis=1):
#                 print(self._aux(quanta, command_batch[:, :-1]).shape)
#                 return
#
#     @tf.function
#     def score(self,batch_quantized_dist):   # (batch_size,quantization_dim,dim+1)
#         batch_size,quantization_dim,dimp1 = batch_quantized_dist.shape
#         batch_quantized_dist_no_coord = batch_quantized_dist[:,:,:self.channel_kernel_dim]
#         weighted_command_batch = self.channel(batch_quantized_dist_no_coord) # (data_batch_size,command_batch_size,command_dim+1)
#         A = [
#             # (batch_size,quantization_dim,1) -> (batch_size,1)
#
#         ]
#         for command in tf.unstack(weighted_command_batch[:, :, :-1], axis=1):
#             batch_quantized_dist_flat =  tf.reshape(batch_quantized_dist, (-1, batch_quantized_dist.shape[-1]))  #(None*quant_dim,dim+1)
#             command_dim=command.shape[1]
#             x = batch_quantized_dist[:,:,:1]*0+self.reshape_command(command) # (None,quant_dim,command_dim)
#             command_batch_flat = tf.reshape(x,(-1,command_dim))
#             a=tf.reduce_sum(tf.reshape(self._aux(batch_quantized_dist_flat, command_batch_flat),(-1,quantization_dim,1)),axis=1)
#             A.append(a)
#         scores = self.concat_total(A)  #(batch_size,command_dim)
#         weight_batch = weighted_command_batch[:, :, -1]
#         return self.dot([scores,weight_batch])
#
#
#
#
#     # @tf.function
#     def train_step(self, batch_quantized_dist):
#         trainable_var = self.trainable_variables
#         with tf.GradientTape() as tape:
#             scores = self.score(batch_quantized_dist)
#             loss = tf.reduce_mean(scores)
#         grad = tape.gradient(loss, trainable_var)
#         self.optimizer.apply_gradients(zip(grad, trainable_var))
#         return {'loss':loss}
