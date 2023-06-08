import numpy as np
import DatasetManager
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sklearn.datasets as skd
from torchdiffeq import odeint_adjoint as odeint
from ffjord import FFJORDModel

FOLDER ='test7'
#__HP__
DATASET_SIZE = 1024*3
BATCH_SIZE = 256
SAMPLE_SIZE = DATASET_SIZE



# _________________________________ODE CONTROL MLP HYPER PARAMETER_________________________________
STACKED_FFJORDS = 2
NUM_HIDDEN = 16
NUM_LAYERS = 2
DATA_DIM = 2

# _________________________________KERNEL HYPER PARAMETER_________________________________

CHANNEL_SIZE = 2
COMMAND_DEPTH = 2
COMMAND_WIDTH = 10

# _________________________________TRAINING HYPER PARAMETER_________________________________
LR = 1e-2
NUM_EPOCHS = 500
THRESHOLD = 0.99

# _________________________________DATASET CHOICE_________________________________
alpha=np.pi/2; R=np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
dataset_as_tensor = np.concatenate([
    # skd.make_circles(n_samples=DATASET_SIZE,noise=.01,factor=0.6)[0]*1,
    # skd.make_circles(n_samples=DATASET_SIZE,noise=.02)[0]*1.5,
    skd.make_moons(n_samples=DATASET_SIZE, noise=.06)[0],
])

SAVE_FOLDER = os.path.join('images',FOLDER)
os.makedirs(SAVE_FOLDER, exist_ok=True)
vizKWarg = {
    'base_folder': '',
    'save_folder': SAVE_FOLDER
}

class MakeMoonsDataset(Dataset):
    def __init__(self, transform=None):
        self.data, self.labels = skd.make_moons(n_samples=DATASET_SIZE, noise=.06, random_state=42)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

class ChannelKernel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChannelKernel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

STACKED_FFJORDS = 2

def flow_gen(ffjord_depth = STACKED_FFJORDS):
    solver = odeint
    bijectors = []
    for _ in range(ffjord_depth):
        mlp_ode_model = FFJORDModel(input_dim=2, hidden_dim=16, output_dim=2)
        bijectors.append(mlp_ode_model)
    return bijectors

flow_family = [flow_gen(ffjord_depth=STACKED_FFJORDS) for _ in range(CHANNEL_SIZE)]
chanel_kernel = ChannelKernel(2, COMMAND_WIDTH, CHANNEL_SIZE)



from MultiFlow import MultiFlow



Moons_Dataset = MakeMoonsDataset()
dataloader = DataLoader(Moons_Dataset, batch_size=32, shuffle=True)
model = MultiFlow(flow_family, chanel_kernel)
for epoch in range(NUM_EPOCHS):
    for batch_data, _ in dataloader:
        loss1, loss2 = model.train()
    pass
