import torch
from torch import nn
from gpytorch.kernels import RBFKernel, ScaleKernel, ConstantKernel

class CustomRBFKernel(nn.Module):
    """
    RBF kernel for Gaussian processes.
    """
    def __init__(self, amplitude=0.0, length_scale=0.0):
        super(CustomRBFKernel, self).__init__()
        self._amplitude = nn.Parameter(torch.tensor(amplitude))
        self._length_scale = nn.Parameter(torch.tensor(length_scale))

    def forward(self, x):
        # Not used for direct computations; keeps parameters structured
        return x

    @property
    def kernel(self):
        # Transformation to ensure positivity during training i.e. softplus(x) = log(1 + exp(x))
        amplitude = torch.nn.functional.softplus(0.1*self._amplitude)
        length_scale = torch.nn.functional.softplus(10 * self._length_scale)

        base_kernel = RBFKernel()
        base_kernel.lengthscale = length_scale
        # print(base_kernel.raw_lengthscale, base_kernel.lengthscale)
        scaled_kernel = ScaleKernel(base_kernel)
        scaled_kernel.outputscale = amplitude
        # print(scaled_kernel.raw_outputscale, scaled_kernel.outputscale)
        kernel = scaled_kernel + ConstantKernel()

        return kernel
    