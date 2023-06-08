

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torch.distributions as D
# D.transforms.AffineTransform
import torch.distributions.transforms as T

# class FFJORDTransform(nn.Module, T.Transform):
class FFJORDTransform(T.Transform):
# class FFJORDTransform(T.AffineTransform):
    def __init__(self, ode_func, T=1.0, timesteps=10):
        super(FFJORDTransform, self).__init__()
        self.ode_func = ode_func
        self.T = T
        self.timesteps = timesteps

    def forward(self, x):
        z, log_det_jacobian = self.ode_func(x, self.T, self.timesteps)
        return z, log_det_jacobian

    def inverse(self, z):
        return z

    def log_abs_det_jacobian(self, x, z):
        _, log_det_jacobian = self.ode_func(x, self.T, self.timesteps)
        return log_det_jacobian

    def domain(self):
        pass

    def codomain(self):
        pass

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2, num_transforms=2):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, t, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.tanh(x)
        # for _ in range(self.num_blocks):
        #     h = torch.relu(self.fc2(h))
        return x


class FFJORDModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=2, num_transforms=2):
        super(FFJORDModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.num_transforms = num_transforms

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

        self.nfe = 0 # Number of function evaluations
        self.ode_func = MLP(input_dim, hidden_dim, output_dim, num_blocks=2, num_transforms=2)

    def forward(self, x):
        self.nfe = 0
        out = odeint(self.ode_func, x, torch.tensor([0, 1], dtype=torch.float32), method='dopri5')
        return out[0], out[1]

    # def ode_func(self, t, x):
    #     self.nfe += 1
    #     out = x
    #     # for _ in range(self.num_transforms):
    #     out = self.transform_block(out)
    #     return out

    def transform_block(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.tanh(x)
        # for _ in range(self.num_blocks):
        #     h = torch.relu(self.fc2(h))
        return x

    def trace_jacobian(self, x):
        with torch.no_grad():
            out, vjp_fn = odeint(self.ode_func, x, torch.tensor([0, 1]), method='dopri5', return_vjp=True)
            vjp = vjp_fn(torch.ones_like(out))
            trace = torch.sum(vjp, dim=-1)
        return trace

# model = FFJORDModel(input_dim=10, hidden_dim=256, output_dim=10, num_blocks=4, num_transforms=16)
# input_data = torch.randn(10, 10)
# output = model(input_data)
# print(output)
