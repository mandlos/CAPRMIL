import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ConstantKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import SmoothedBoxPrior
from linear_operator.operators import DiagLinearOperator
from gpytorch.utils.memoize import cached

# Variational strategy where prior can change
class CustomVariationalStrategy(VariationalStrategy):
    def __init__(self, *args, prior_mean_value=0.0, prior_var_value=1.0, **kwargs):
        """
        Custom Variational Strategy that allows modifying the prior distribution.
        
        Args:
            prior_mean_value (float): Mean of the prior distribution (default 0.0).
            prior_var_value (float): Variance (diagonal elements) of the prior covariance matrix (default 1.0).
        """
        super().__init__(*args, **kwargs)
        self.prior_mean_value = prior_mean_value
        self.prior_var_value = prior_var_value

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.full(
            self._variational_distribution.shape(),
            fill_value=self.prior_mean_value,
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        diag_elements = torch.full_like(zeros, fill_value=self.prior_var_value)
        return MultivariateNormal(zeros, DiagLinearOperator(diag_elements))

# Sparse gaussian process layer that uses custom variational strategy
class gp_layer(ApproximateGP):
    '''
    Variational Gaussian Process layer
    '''
    def __init__(self, num_inducing_points_z, inducing_points_dimensions_Z, jitter:float=1.e-6, prior_mean=0, prior_variance=1., mean='linear'):
        assert isinstance(num_inducing_points_z, int), 'Number of inducing points must be an integer'
        assert isinstance(inducing_points_dimensions_Z, int), 'Number of inducing points dimensions must be an integer'

        # Initialize inducing point locations Z
        inducing_points_z = 0. + (1. - 0.) * torch.rand(num_inducing_points_z, inducing_points_dimensions_Z)

        # Prior of inducing point q(u)
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=num_inducing_points_z, mean_init_std=1e-3)

        # Strategy to calculate prior of f|u from prior of u
        variational_strategy = CustomVariationalStrategy(self, inducing_points_z, 
                                                         variational_distribution=variational_distribution, 
                                                         learn_inducing_locations=True, jitter_val=jitter, 
                                                         prior_mean_value=prior_mean, prior_var_value=prior_variance)
        
        super().__init__(variational_strategy)
        
        # Predictive mean on actual training/testing points
        if mean == 'linear':
            self.mean_module = LinearMean(input_size=inducing_points_dimensions_Z)
        elif mean == 'constant': 
            self.mean_module = ConstantMean()
            # self.mean_module.initialize(constant=0.)
        else:
            raise ValueError(f"Unsupported mean type: {mean}")

        lengthscale_prior = None
        # lengthscale_prior = SmoothedBoxPrior(1.e-2, 1., sigma=0.01, transform=None)
        self.rbf_kernel = RBFKernel(lengthscale_prior=lengthscale_prior)

        outputscale_prior = None
        # outputscale_prior = SmoothedBoxPrior(1.e-2, 1., sigma=0.01, transform=None)

        self.scale_kernel = ScaleKernel(base_kernel=self.rbf_kernel, outputscale_prior=outputscale_prior)
        self.cov_module = self.scale_kernel + ConstantKernel()   # Vanilla RBF kernel for the GP f|u


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.cov_module(x)
        return MultivariateNormal(mean_x, covar_x)

class SGPMIL(nn.Module):
    '''
    Attention mechanism for MIL with Gaussian Process.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dims = self.config['data']['data_dims']
        self.hl1 = self.config['model']['hidden_layer_size_1'] != 0
        self.hl2 = self.config['model']['hidden_layer_size_2'] != 0
        
        if self.config['model']['attn_hl_activation'] == 'sigmoid':
            self.att_hl_activation = nn.Sigmoid()
        else:
            self.att_hl_activation = nn.Tanh()

        self.post_att_activation = self.config['model']['post_attention_activation']
        self.att_type = self.config['model']['attention']
        self.att_multiplication_type = self.config['model']['attention_multiplication_type']
        self.num_inducing_points = self.config['model']['inducing_points']
        self.inducing_point_dims = self.config['model']['hidden_layer_size_att']
        self.num_classes = self.config['data']['num_classes']
        self.mc_samples = self.config['model']['mc_samples']

        self._assertions()
        # Definition of model layers
        layer_list = [nn.Linear(in_features=self.data_dims, out_features=self.config['model']['hidden_layer_size_0']),
                      nn.ReLU()]
        att_in_features = self.config['model']['hidden_layer_size_0']

        if self.hl1:
            layer_list.extend([nn.Linear(in_features=self.config['model']['hidden_layer_size_0'], out_features=self.config['model']['hidden_layer_size_1']),
                              nn.ReLU()])
            att_in_features = self.config['model']['hidden_layer_size_1']

        if self.hl2:
            layer_list.extend([nn.Linear(in_features=att_in_features, out_features=self.config['model']['hidden_layer_size_2']),
                              nn.ReLU()])
            att_in_features = self.config['model']['hidden_layer_size_2']

        self.mlp = nn.Sequential(*layer_list)

        # Attention gp block      
        self.sgp = gp_layer(num_inducing_points_z=self.num_inducing_points,
                            inducing_points_dimensions_Z=self.inducing_point_dims,
                            jitter=self.config['model']['jitter'], 
                            prior_mean=self.config['model']['prior_mean'], 
                            prior_variance=self.config['model']['prior_variance'])
        
        layers = [nn.Linear(in_features=att_in_features, out_features=self.inducing_point_dims),
                  self.att_hl_activation]

        layers.append(self.sgp)
        self.f_gp = nn.Sequential(*layers)

        # Final classification layer without softmax
        self.cl = nn.Linear(in_features=att_in_features, out_features=self.num_classes)
        # Activation of the final classification layer
        self.act = nn.Softmax(dim=-1)

    def _assertions(self):
        assert self.att_type in ['sgpmil'], 'Only GP attention implemented currently'
        assert self.att_multiplication_type in ['elementwise'], 'Only elementwise or dot attention multiplication supported'
        assert self.post_att_activation in ['softmax', 'sigmoid'], 'Only softmax and sigmoid activations for attention are supported'
        assert isinstance(self.att_hl_activation, (nn.Sigmoid, nn.Tanh)), "s must be either nn.Sigmoid or nn.Tanh"

    def mc_sampling(self, x):
        '''
        Monte carlo sampling of GP output distribution. Use reparameterization scheme mean(x) + cov(x) * eps, eps ~ N(0, 1) for backpropagation.
        Args:
            x: GP output distribution
        Returns:
            samples: samples from the GP output distribution of shape [samples, batch_size, Num_of_datapoints]
        '''
        mc_samples = torch.Size([self.mc_samples])
        if self.config['model']['sampling'] == 'cov':
            samples = x.rsample(mc_samples)
        elif self.config['model']['sampling'] == 'var':
            std = x.variance.unsqueeze(dim=0).sqrt()
            epsilon = torch.randn(self.config['model']['mc_samples'], std.shape[0], device = self.device)
            mean = x.mean.unsqueeze(dim=0).to(self.device)
            return (mean + std * epsilon).squeeze(dim=0).unsqueeze(dim=-1)
        else:
            raise ValueError(f"Unsupported sampling method: {self.config['model']['sampling']}")
        return samples

    def mc_integration(self, x, num_classes):
        x_mean = torch.mean(x,  dim=1).view(1, num_classes)
        x_std = torch.std(x.squeeze(dim=0), axis=0)
        return x_mean, x_std

    def custom_activation(self, x: torch.Tensor = None, mc_samples: int = 1):    
        '''
        Softmax for the GP output to interpret 'em as attention
        Args:
            x: GP output samples with shape like [MC_samples, batch_size, Num_of_datapoints]
        Returns:
            activation(x): Activation of the input x with shape [batch_size, MC_samples, Num_of_datapoints]
        '''
        assert x is not None, "Input tensor x is required"

        x = x.view(mc_samples, 1, -1)  # Reshape to add an axis for MC samples
        if self.post_att_activation == 'softmax':
            x = F.softmax(x, dim=-1)  # Apply softmax along the last dimension
        elif self.post_att_activation == 'sigmoid':
            x = F.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.post_att_activation}")

        out = x.view(mc_samples, -1)
        return out
        
    def attention_multiplication(self, i):
        '''
        Multiply features with respective attention weights.
        Args:
            i: List containing attention weights and features i.e. [attention_weights, features]
        Returns:
            out: Attention weighted features i.e. slide-level representation
        '''
        a = i[0]
        f = i[1]
        out = a.unsqueeze(-1) * f
        out = out.unsqueeze(0) 
        return out

    def reshape_pre_mc_integration(self, x, mc_samples, num_classes):
        '''
        Reshape the output of the classification layer [1, MC_samples, Num_classes]
        '''
        return x.view(1, mc_samples, num_classes)

    def get_slide_representation(self, a, x):
        x = self.attention_multiplication([a, x])
        
        if x.dim() == 3:
            x = x.sum(dim=1)
        elif x.dim() == 4:
            x = x.sum(dim=2)
        else:
            raise ValueError('Invalid input dimensions')
        return x
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.mlp(x)

        # Variational q(A) & slide representation samples
        f_gp = self.f_gp(x)
        f = self.mc_sampling(f_gp)
        a = self.custom_activation(x=f, mc_samples=self.mc_samples)
        x = self.get_slide_representation(a, x)
        
        # Classifier
        x_logits = self.cl(x)   
        x = self.act(x_logits)

        x_pre = x.view(1, self.mc_samples, self.num_classes)
        x_logits = x_logits.view(1, self.mc_samples, self.num_classes)
        
        # MC integration
        x, x_std = self.mc_integration(x_pre, self.num_classes)    
        x_logits, _ = self.mc_integration(x_logits, self.num_classes)
        
        results = {'y_hat': x, 'y_hat_se': x_std, 
                   'logits': x_logits, 'attention':a, 
                   'pre_mc_integration': x_pre}

        return results

class AGP(nn.Module):
    def __init__(self, config):
        super(AGP, self).__init__()
        self.config = config
        self.data_dims = self.config['data']['data_dims']

        layers = []
        hidden_sizes = [
            self.config['model']['hidden_layer_size_0'],
            self.config['model'].get('hidden_layer_size_1', 0),
            self.config['model'].get('hidden_layer_size_2', 0)]

        input_dim = self.data_dims
        for h_size in hidden_sizes:
            if h_size > 0:
                layers.append(nn.Linear(input_dim, h_size))
                layers.append(nn.ReLU())
                input_dim = h_size

        self.mlp = nn.Sequential(*layers)
        
        self.pre_sgp = nn.Sequential(nn.Linear(in_features=input_dim, out_features=self.config['model']['hidden_layer_size_att']), 
                                     nn.Sigmoid())
        self.sgp = gp_layer(num_inducing_points_z=self.config['model']['inducing_points'],
                            inducing_points_dimensions_Z=self.config['model']['hidden_layer_size_att'],
                            jitter=self.config['model']['jitter'], 
                            prior_mean=self.config['model']['prior_mean'], 
                            prior_variance=self.config['model']['prior_variance'],
                            mean='constant')

        self.sgp_postact = nn.Softmax(dim=-1)
        # self.sgp_postact = nn.Sigmoid()

        self.classifier = nn.Linear(in_features=input_dim, out_features=self.config['data']['num_classes'])
        self.activation = nn.Softmax(dim=-1)

    def sampling(self, f):
        mc_samples = torch.Size([self.config['model']['mc_samples']])
        return f.rsample(mc_samples)

    def pooling(self, a, x):
        return self.attention_multiplication(a, x)

    def attention_multiplication(self, a, x):
        return torch.matmul(a, x)

    def integration(self, x):
        x_mean = torch.mean(x, dim=1).view(1, self.config['data']['num_classes'])
        x_std = torch.std(x, dim=1).view(1, self.config['data']['num_classes'])
        return x_mean, x_std

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        h = self.mlp(x)
        h_pre = self.pre_sgp(h)
        f_mvn = checkpoint(self.sgp, h_pre)
        f_mvn.loc = torch.clamp(f_mvn.loc, min=1.e-8)  # avoid NaNs in KL term
        a_post = self.sampling(f_mvn)
        a = self.sgp_postact(a_post)
        slide_samples = self.pooling(a, h)
        logits = self.classifier(slide_samples).view(1, self.config['model']['mc_samples'], self.config['data']['num_classes'])
        probs = self.activation(logits) 
        logits_mean, logits_std = self.integration(logits)
        probs_mean, probs_std = self.integration(probs) 
        return {'y_hat': probs_mean, 'y_hat_std': probs_std, 'logits': logits_mean, 'logits_std': logits_std, 'attention': a}