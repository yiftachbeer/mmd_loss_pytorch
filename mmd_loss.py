import torch
from torch import nn


class MMDLoss(nn.Module):

    def __init__(self, kernel_mul=2.0, kernel_num=5, bandwidth=None):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.bandwidth = bandwidth

    def gaussian_kernel(self, X, Y):
        total = torch.vstack([X, Y])
        L2_distances = torch.cdist(total, total) ** 2

        n_samples = X.shape[0] + Y.shape[0]
        bandwidth = self.bandwidth if self.bandwidth is not None else torch.sum(L2_distances.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = bandwidth * self.kernel_mul ** torch.arange(self.kernel_num)

        return torch.exp(-L2_distances[None, ...] / bandwidth_list[:, None, None]).sum(dim=0)

    def forward(self, X, Y):
        kernels = self.gaussian_kernel(X, Y)

        X_size = X.shape[0]
        XX = kernels[:X_size, :X_size]
        YY = kernels[X_size:, X_size:]
        XY = kernels[:X_size, X_size:]
        return torch.mean(XX - XY - XY.T + YY)