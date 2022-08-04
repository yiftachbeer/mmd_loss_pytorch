import torch
from torch import nn


class RBF(nn.Module):

    def __init__(self, kernel_mul=2.0, kernel_num=5, bandwidth=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.bandwidth = bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2

        n_samples = len(X)
        bandwidth = L2_distances.data.sum() / (n_samples ** 2 - n_samples) if self.bandwidth is None else self.bandwidth
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = bandwidth * self.kernel_mul ** torch.arange(self.kernel_num)

        return torch.exp(-L2_distances[None, ...] / bandwidth_list[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY