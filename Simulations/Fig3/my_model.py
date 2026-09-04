import torch
import torch.nn as nn
import numpy as np


def to_onehot(target):
    # change the target to one-hot version
    Y = np.ravel(target.numpy()).astype(int)
    Y_train = np.zeros((Y.shape[0], Y.max()-Y.min()+1))
    Y_train[np.arange(Y.size), Y-Y.min()] = 1
    target_onehot =torch.from_numpy(Y_train.astype(np.float32))
    return target_onehot


class Discriminator(nn.Module):
    def __init__(self, ndim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ndim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1)
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, X):
        validity = self.model(X)
        return validity
    
class Generator(nn.Module):
    def __init__(self, xdim, ndim,outdim=2):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(xdim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, ndim),
        )
        self.fc = nn.Sequential(
            nn.Linear(ndim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, outdim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, X):
        latent = self.model(X)
        out = self.fc(latent)
        return latent, out

class pred_from_Rx(nn.Module):
    def __init__(self, ndim,outdim=2):
        super(pred_from_Rx, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(ndim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, outdim),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, X):
        out = self.fc(X)
        return out
