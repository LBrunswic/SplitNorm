import tensorflow as tf
import tensorflow_probability as tfp
from time import time
tfb = tfp.bijectors
tfd = tfp.distributions
import numpy as np


class MLP_ODE(tf.keras.Model):
    def __init__(self, kernel_gen, kernelKWarg={}, name='mlp_ode'):
        super(MLP_ODE, self).__init__()
        self.kernel_gen = kernel_gen
        self.kernelKWarg = kernelKWarg
        self.kernel = self.kernel_gen(**self.kernelKWarg)
    def call(self, t, inputs):
        inputs = tf.concat([tf.broadcast_to(t, inputs.shape), inputs], -1)
        return self.kernel(inputs)

    @classmethod
    def sequential_dense(self,num_hidden,num_layers,num_output,kernelKWarg={}):
        def aux_gen():
            layers = [
                tf.keras.layers.Dense(num_hidden,**kernelKWarg)
                for i in range(num_layers-1)
            ]
            layers.append(tf.keras.layers.Dense(num_output,**kernelKWarg))
            return tf.keras.Sequential(layers)
        return self(aux_gen)


def flow_gen(mlp_model, DATA_DIM=2,ffjord_depth = 2,ODEKernelarg=[],ODEKernelKwarg={}):
    solver = tfp.math.ode.DormandPrince(atol=1e-5)
    ode_solve_fn = solver.solve
    # trace_augmentation_fn = tfb.ffjord.trace_jacobian_exact
    trace_augmentation_fn = tfb.ffjord.trace_jacobian_hutchinson
    bijectors = []
    for _ in range(ffjord_depth):
        bijectors.append(tfp.bijectors.Shift(tf.Variable([0.]*DATA_DIM,trainable=True)))
        # bijectors.append(tfp.bijectors.ScaleMatvecTriL(scale_tril=tf.Variable([[1., 0.], [0, 1.]])))
        next_ffjord = tfb.FFJORD(
            state_time_derivative_fn=mlp_model(*ODEKernelarg,kernelKWarg=ODEKernelKwarg),
            ode_solve_fn=ode_solve_fn,
            trace_augmentation_fn=trace_augmentation_fn
        )
        bijectors.append(next_ffjord)
    return tfb.Chain(bijectors[::-1])
