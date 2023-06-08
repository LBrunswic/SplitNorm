import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from torch.distributions import MultivariateNormal
import torch.optim as optim
import torch.distributions as distributions
from ffjord import FFJORDTransform

def saturate(x, threshold=0.9, dtype=torch.float32):
    s = x > threshold
    COMMAND_SIZE = s.shape[-1]
    print(COMMAND_SIZE)
    sb = ~s
    p = torch.empty(COMMAND_SIZE, dtype=torch.bool)
    for i in range(COMMAND_SIZE):
        pi = torch.ones_like(x[:, 0], dtype=torch.bool)
        for j in range(COMMAND_SIZE):
            if j != i:
                pi = pi & sb[:, j]
                p[i] = pi
    p = p.T.to(dtype)
    s = s.to(dtype)
    sb = sb.to(dtype)
    return sb * p * x + s

class MultiFlow(nn.Module):
    def __init__(self, flow_family, channel_kernel):
        super(MultiFlow, self).__init__()
        self.channel_size = len(flow_family)
        self.flow_family = flow_family
        self.channel_kernel = channel_kernel

        # Distribution KWarg
        mu = torch.zeros(self.channel_size)
        variance = torch.ones(self.channel_size)

        self.base_distribution_gen = MultivariateNormal
        self.base_distribution = self.base_distribution_gen(mu, torch.diag(variance))

        self.flow_distribution = []
        for i in range(self.channel_size):
            self.flow_distribution.append(FFJORDTransform(self.flow_family[i]))

        self.transformed_distributions = [distributions.TransformedDistribution(self.base_distribution, F) for F in self.flow_distribution]

        learning_rate = torch.tensor(0.01, requires_grad=False)
        NF_parameters = []
        for kernel in self.flow_family:
            NF_parameters.append(kernel.parameters())
        self.optimizer1 = optim.Adam(NF_parameters, lr=0.01)
        self.optimizer2 = optim.Adam(self.channel_kernel.parameters(), lr=0.01)

    def forward(self, inputs, threshold=0.6):
        C = self.channel_kernel(inputs)
        C_saturated = saturate(C)
        images_intensity = torch.stack(
            [-tds.log_prob(inputs) for tds in self.transformed_distributions]
        )
        return C, C_saturated, images_intensity
        # return 0

    def train_step(self, target_sample, threshold=0.6):
        # train the NF
        for F in self.flow_family:
            F.train()
        _, FinalC, images_intensity = self(target_sample,threshold=threshold)
        loss1 = torch.mean(torch.sum(FinalC * images_intensity, dim=1))
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        # train the channel kernel
        self.channel_kernel.train()
        C, _, images_intensity = self(target_sample, threshold=threshold)
        loss2 = torch.mean(torch.sum(C * images_intensity, dim=1))
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        pass