import tensorflow as tf

class CommandedMLP_ODE(tf.keras.Model):
    def __init__(self, kernel, command_dim=0, name='mlp_ode'):
        super(CommandedMLP_ODE, self).__init__()
        self.kernel = kernel
        self.command_dim = command_dim

    @tf.function
    def call(self, t, x,command=None):
        t_x = tf.concat([t+0*x[:,:1], x], -1)
        print(t_x.shape)
        return self.kernel((t_x,command))
