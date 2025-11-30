# --- Standard Library ---
from typing import Dict

# --- Third Party ---
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
import torchmetrics
import gpytorch
from sklearn.metrics import balanced_accuracy_score
import wandb

# --- Custom/Local Modules ---
from topk.svm import SmoothTop1SVM
from utils.utils import DictToAttr
from MyOptimizer.optim_factory import create_optimizer
from schedulers.LinearCosine import LinearWarmupCosineAnnealingLR
import src
from src.model_abmil import ABMIL
from src.model_clam import CLAM_SB, CLAM_MB
from src.model_transmil import TransMIL
from src.model_dgrmil import DGRMIL
from src.model_bmil import probabilistic_MIL_Bayes_spvis as BMIL_spvis
from src.model_bmil_subtyping import probabilistic_MIL_Bayes_spvis as BMIL_spvis_subtyping
from src.model_bmil import get_ard_reg_vdo
from src.model_tsmil import TSMIL
from src.gp_models import SGPMIL, AGP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
# Pytorch lightning wrapper for models
# Balanced accuracy: average of recall obtained on each class, for binary actualy TPR + TNR / 2

class LitGPModel(pl.LightningModule):
    def __init__(self, config: Dict={}, num_training_points: int=0, num_val_points: int=0):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        self.print_model = self.config['model']['print_model']
        
        self.model = self._init_model()
        self.num_training_points = num_training_points
        self.num_val_points = num_val_points

        self.num_classes = self.config['data']['num_classes']
        self._init_metrics()

        # Losses separately
        losses = ['train_ce', 'train_kl', 'val_ce', 'val_kl']
        for loss in losses:
            setattr(self, loss, [])

        # Predictions and targets
        output_variables = ['train_preds', 'train_targets', 'val_preds', 'val_targets', 'test_preds', 'test_targets']
        for var in output_variables:
            setattr(self, var, [])

        self.first_low_lr_epoch = None

    def _assertions(self):
        if self.config['phase'] == 'train':
            assert self.model is not None, 'Model not provided'
        assert self.config, 'Config empty'
        assert self.num_training_points > 0, 'No training points provided'
        assert self.num_val_points > 0, 'No validation points provided'

    def setup(self, stage: str=None):
        wandb.define_metric('Epoch')
        wandb.define_metric('train/*', step_metric='Epoch')
        wandb.define_metric('val/*', step_metric='Epoch')
        wandb.define_metric('test/*', step_metric='Epoch')
        if stage == 'fit':
            self._assertions()
            print(f"Training points: {self.num_training_points}, Validation points: {self.num_val_points}")

    def _init_metrics(self):
        # Initialize metrics
        average = 'weighted'
        task = 'binary' if self.num_classes == 2 else 'multiclass'

        self.train_accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average=average).to(self.device)

        self.val_accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average=average).to(self.device)
        self.val_auc = torchmetrics.AUROC(task, num_classes=self.num_classes).to(self.device)
        self.val_cohenkappa = torchmetrics.CohenKappa(task=task, num_classes=self.num_classes, weights='quadratic').to(self.device)

        self.test_accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average=average).to(self.device)
        self.test_auc = torchmetrics.AUROC(task, num_classes=self.num_classes).to(self.device)
        self.test_cohenkappa = torchmetrics.CohenKappa(task=task, num_classes=self.num_classes, weights='quadratic').to(self.device)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task=task, num_classes=self.num_classes).to(self.device)
        self.test_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').to(self.device)

    def _init_model(self):
        if self.config['model']['attention'] == 'sgpmil':
            return SGPMIL(self.config)
        elif self.config['model']['attention'] == 'agp':
            return AGP(self.config)
        else:
            raise ValueError('Model not supported')
    
    def forward(self, x, y):
        if self.config['model']['attention'] == 'sgpmil':
            return self.model(x.unsqueeze(dim=0))
        elif self.config['model']['attention'] == 'agp':
            return self.model(x.unsqueeze(dim=0))
        else:
            raise ValueError('Attention mechanism not supported')

    def on_train_start(self) -> None:
        # Print the model structure
        if self.print_model: print(self)

    def training_step(self, batch, batch_idx):
        # Data
        x, y = batch['img'], batch['label']
        # Model forward
        raw_out = self.forward(x, y)
        y_hat = raw_out['y_hat']
        
        # Compute loss
        ce, kl = self.compute_loss(raw_out, y, self.num_training_points)
        loss = ce + kl

        self.train_ce.append(ce)
        self.train_kl.append(kl)
        self.train_accuracy.update(y_hat.argmax(dim=1), y)
        return loss
    
    def on_train_epoch_end(self):
        # Compute the average train loss
        avg_ce = torch.stack(self.train_ce).mean()
        avg_kl = torch.stack(self.train_kl).mean()
        avg_loss = avg_ce + avg_kl

        wandb.log({
            'Epoch': self.current_epoch,
            'train/loss':{
                'train/ce': avg_ce.item(),
                'train/kl': avg_kl.item(),
                },
            'train/avg_loss': avg_loss.item(),
            'train/accuracy': self.train_accuracy.compute().item(),
        })

        self.train_accuracy.reset()
        self.train_ce.clear()
        self.train_kl.clear()
     
    def validation_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']

        raw_out = self.forward(x,y)
        y_hat = raw_out['y_hat']

        # Compute loss
        ce, kl = self.compute_loss(raw_out, y, self.num_val_points)
        loss = ce + kl
        
        self.val_ce.append(ce)
        self.val_kl.append(kl)
        self.val_accuracy.update(y_hat.argmax(dim=1), y)
        self.val_auc.update(y_hat, y) if self.num_classes > 2 else self.val_auc.update(y_hat[:, 1], y)
        self.val_cohenkappa.update(y_hat, y) if self.num_classes > 2 else self.val_cohenkappa.update(y_hat[:, 1], y)
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))

        self.val_preds.append(y_hat.clone().detach())
        self.val_targets.append(y.clone().detach())
        return loss
    
    def on_validation_epoch_end(self):
        avg_ce = torch.stack(self.val_ce).mean()
        avg_kl = torch.stack(self.val_kl).mean()
        avg_loss = avg_ce + avg_kl

        y_true = torch.cat(self.val_targets, dim=0).cpu()
        y_pred = torch.stack(self.val_preds, dim=0).cpu().argmax(dim=-1)
        
        self.log('val/balanced_accuracy', balanced_accuracy_score(y_true=y_true, y_pred=y_pred))

        wandb.log({'Epoch':self.current_epoch,
                    'val/loss':{
                        'val/ce': avg_ce.item(),
                        'val/kl': avg_kl.item(),
                    }, 
                    'val/avg_loss': avg_loss.item(),
                    'val/accuracy': self.val_accuracy.compute().item(),
                    'val/auc': self.val_auc.compute().item(),
                    'val/cohenskappa': self.val_cohenkappa.compute().item(),
                })

        if self.current_epoch != (self.trainer.max_epochs - 1) and self.val_preds and self.val_targets:
            self.val_preds.clear()
            self.val_targets.clear()

        self.val_ce.clear()
        self.val_kl.clear()
        self.val_accuracy.reset()
        self.val_auc.reset()
        self.val_cohenkappa.reset()

    def on_train_end(self):
        self.val_preds.clear()
        self.val_targets.clear()

        pass

    def test_step(self, batch, batch_idx):
        x, y = batch['img'], batch['label']    
        raw_out = self.forward(x,y)
        y_hat = raw_out['y_hat']

        # Metrics
        self.test_accuracy.update(y_hat.argmax(dim=1), y)
        self.test_auc.update(y_hat, y) if self.num_classes > 2 else self.test_auc.update(y_hat[:, 1], y)
        self.test_cohenkappa.update(y_hat, y) if self.num_classes > 2 else self.test_cohenkappa.update(y_hat[:, 1], y)
        self.test_confusion_matrix.update(y_hat, y) if self.num_classes > 2 else self.test_confusion_matrix.update(y_hat[:, 1], y)
        self.test_ece.update(y_hat, y) if self.num_classes > 2 else self.test_ece.update(y_hat[:, 1], y)

        self.test_preds.append(y_hat)
        self.test_targets.append(y)
        pass

    def on_test_epoch_end(self):
        wandb.log({'test/accuracy': self.test_accuracy.compute().item(),
                   'test/balanced_accuracy': balanced_accuracy_score(y_true=torch.cat(self.test_targets, dim=0).cpu(), 
                                                                     y_pred=torch.stack(self.test_preds, dim=0).cpu().argmax(dim=-1)),
                  'test/auc': self.test_auc.compute().item(),
                  'test/cohenkappa': self.test_cohenkappa.compute().item(), 
                  'test/ece': self.test_ece.compute().item()
                  })
        wandb.log({'test/confusion_matrix': wandb.Image(self.test_confusion_matrix.plot()[0]),
                   'test/ece_curve': wandb.Image(self.test_ece.plot()[0])})

        self.test_accuracy.reset()
        self.test_auc.reset()
        self.test_cohenkappa.reset()
        self.test_confusion_matrix.reset()
        self.test_ece.reset()
        pass

    def compute_loss(self, raw_out, y, num_points=1):
        # KL for posterior|prior of variational inducing posterior q(u)
        kl = self.variational_loss(num_points, raw_out, y)
        # Instead of maximizing the log-likelihood, minimize a standard cross-entropy (shown to be equivalent)
        ll = CrossEntropyLoss()(raw_out['logits'], y)
        return ll, kl

    def variational_loss(self, num_points, raw_out, y):
        _kl_factor = self.config['model']['kl_factor']
        layer = self.model.sgp         
        _kl_weight = torch.tensor(_kl_factor / num_points, dtype=torch.float32, device=self.device)
        _kl_div = self.compute_kl_divergence(layer)
        _kl_loss = _kl_weight * _kl_div
        return _kl_loss
    
    def compute_kl_divergence(self, layer):
        """
        Compute KL divergence between two distributions q and p.
        q: variational distribution---> MultiVariateNormal
        p: prior distribution---> MultiVariateNormal
        """
        return layer.variational_strategy.kl_divergence()
    
    def compute_diff_entropy(self, variable):
        p = variable.flatten()
        return torch.mean(-p*torch.log(p + 1.e-6), dim=-1)

    def configure_optimizers(self):
        # Define optimizer
        assert self.config['training']['optimizer'] in ['adam', 'sgd', 'adamw', 'lookahead_radam'], 'Optimizer not supported, must be adam, sgd or lookahead_radam'
        learning_rate = self.config['training']['learning_rate']
        if self.config['training']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.config['training']['reg'])
        elif self.config['training']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=self.config['training']['reg'])
        elif self.config['training']['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=self.config['training']['reg'], 
                                        momentum=self.config['training']['momentum'])
        elif self.config['training']['optimizer'] == 'lookahead_radam':
            opt_args = {'opt':'lookahead_radam', 'lr':learning_rate, 
                        'opt_eps':self.config['training']['opt_eps'], 
                        'opt_betas':self.config['training']['opt_betas'], 
                        'momentum':self.config['training']['momentum'],
                        'weight_decay':self.config['training']['reg']}
            opt_args_obj = DictToAttr(**opt_args)
            optimizer = create_optimizer(opt_args_obj, self.model)
        else:
            raise ValueError('Optimizer not supported')
        
        # Configure scheduler
        def lr_lambda(epoch):
            decay_after = self.config['training']['lr_decay_after_epoch']
            stop_decay_after = self.config['training']['stop_decay_after_epoch']
            lr_decay_value_stop = self.config['training']['stop_decay_lr_value']
            decay_factor = self.config['training']['lr_decay_factor']
            current_lr = optimizer.param_groups[0]['lr']
        
            if epoch < decay_after:
                return 1.0
            
            if epoch > stop_decay_after or current_lr <= lr_decay_value_stop:
                if self.first_low_lr_epoch is None:
                    self.first_low_lr_epoch = epoch

                return decay_factor ** (self.first_low_lr_epoch - decay_after + 1)
            
            return decay_factor ** (epoch - decay_after + 1)
        
        if self.config['training']['scheduler'] == 'lambda':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif self.config['training']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['training']['max_epochs'],
                                                                   eta_min=self.config['training']['min_lr'])
        elif self.config['training']['scheduler'] == 'linearcosine':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.config['training']['warmup_epochs'],
                                                      max_epochs=self.config['training']['max_epochs'], eta_min=self.config['training']['min_lr'],
                                                      warmup_start_lr=self.config['training']['warmup_lr'])
        lr_scheduler_config = {'scheduler': scheduler, 'interval': self.config['training']['lr_logging_interval'],
                               'frequency': self.config['training']['lr_logging_frequency']}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def _monitor_kernel_hyperparams(self):
        for kernel in self.model.sgp.cov_module.sub_kernels():
            if isinstance(kernel, gpytorch.kernels.ScaleKernel):
                lengthscale = kernel.base_kernel.lengthscale.item() 
                raw_lengthscale = kernel.base_kernel.raw_lengthscale.item()
                outputscale = kernel.outputscale.item()
                raw_outputscale = kernel.raw_outputscale.item()
            elif isinstance(kernel, gpytorch.kernels.ConstantKernel):
                raw_constant = kernel.raw_constant.item()
                constant = kernel.constant.item()
        return {'lengthscale': lengthscale, 'raw_lengthscale': raw_lengthscale, 
                'outputscale': outputscale, 'raw_outputscale': raw_outputscale, 
                'constant': constant, 'raw_constant': raw_constant}


class LitDetModel(pl.LightningModule):
    def __init__(self, config: Dict={}):
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(ignore=['model'])
        self.print_model = self.config['model']['print_model']

        # Returns a dictionary with keys 'logits', 'Y_prob', 'Y_hat', 'A_raw', 'results_dict'
        self.model = self._init_model()
        self.num_classes = self.config['data']['num_classes']
        self._init_metrics()

        # Assertions
        self._assertions()
        
        # Loss separately
        losses = ['train_bag_loss', 'train_instance_loss', 'val_bag_loss', 'val_instance_loss']
        for loss in losses:
            setattr(self, loss, [])

        # Predictions and targets
        output_variables = ['train_preds', 'train_targets', 'val_preds', 'val_targets', 'test_preds', 'test_targets']
        for var in output_variables:
            setattr(self, var, [])

        self.first_low_lr_epoch = None

    def setup(self, stage=None):
        wandb.define_metric('Epoch')
        wandb.define_metric('train/*', step_metric='Epoch')
        wandb.define_metric('val/*', step_metric='Epoch')
        wandb.define_metric('test/*', step_metric='Epoch')
        
    def _init_metrics(self):
        # Initialize metrics
        average = 'weighted'
        task = 'binary' if self.num_classes == 2 else 'multiclass'
        metrics_device = self.device

        self.train_accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average=average).to(metrics_device)
        self.train_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').cpu()

        self.val_accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average=average).to(metrics_device)
        self.val_auc = torchmetrics.AUROC(task, num_classes=self.num_classes).cpu()
        self.val_cohenkappa = torchmetrics.CohenKappa(task=task, num_classes=self.num_classes, weights='quadratic').cpu()
        self.val_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').cpu()

        self.test_accuracy = torchmetrics.Accuracy(task, num_classes=self.num_classes, average=average).to(metrics_device)
        self.test_auc = torchmetrics.AUROC(task, num_classes=self.num_classes).cpu()
        self.test_cohenkappa = torchmetrics.CohenKappa(task=task, num_classes=self.num_classes, weights='quadratic').cpu()
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task=task, num_classes=self.num_classes).cpu()
        self.test_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').cpu()

    def _init_model(self):
        if self.config['model']['attention'] == 'clam':
            instance_loss = SmoothTop1SVM(n_classes=2) if self.config['model']['instance_loss_fn'] == 'svm' else nn.CrossEntropyLoss()
            model_dict = {'gate': self.config['model']['gate'], 'size_arg' : self.config['model']['model_size'], 
                        'dropout' : self.config['model']['dropout'], 'k_sample': self.config['model']['B'], 
                        'n_classes': self.config['data']['num_classes'], 'instance_loss_fn': instance_loss.cuda(device), 
                        'subtyping': self.config['model']['subtyping'], 'embed_dim': self.config['model']['embed_dim'], }
            if self.config['model']['variant'] == 'sb':
                model = CLAM_SB(**model_dict)
            elif self.config['model']['variant'] == 'mb':
                model = CLAM_MB(**model_dict)
        elif self.config['model']['attention'] == 'transmil':
            model = TransMIL(n_classes=self.config['data']['num_classes'], in_size=self.config['model']['embed_dim'])
        elif self.config['model']['attention'] == 'abmil':
            model = ABMIL(in_features=self.config['model']['embed_dim'], n_classes=self.config['data']['num_classes'],
                          gated=self.config['model']['gate'], dropout=self.config['model']['dropout'],
                          attn_branches=self.config['model']['attn_branches'], 
                          L=self.config['model']['hidden_layer1'], M=self.config['model']['hidden_layer2'])
        elif self.config['model']['attention'] == 'dgrmil':
            model = DGRMIL(in_features=self.config['model']['embed_dim'], L=self.config['model']['L'],
                           n_lesion=self.config['model']['num_les'], num_classes=self.config['data']['num_classes'],
                           dropout_node=self.config['model']['dropout'], 
                           dropout_patch=self.config['model']['dropout_patch'])
        elif self.config['model']['attention'].startswith('bayesmil'):
            if self.config['data']['num_classes'] == 2 and not self.config['model']['subtyping']:
                model = BMIL_spvis(gate=self.config['model']['gate'], 
                                   size_arg=self.config['model']['model_size'],
                                   encoder=self.config['model']['feature_extractor'].split('_')[0].lower(),
                                   patch_size=self.config['data']['patch_size'],
                                   dropout=self.config['model']['dropout'],
                                   n_classes=self.config['data']['num_classes'],
                                   top_k=self.config['model']['topk'])
            else:
                print('\n\nSubtyping\n\n')
                model = BMIL_spvis_subtyping(gate=self.config['model']['gate'], 
                                             size_arg=self.config['model']['model_size'],
                                             encoder=self.config['model']['feature_extractor'].split('_')[0].lower(),
                                             patch_size=self.config['data']['patch_size'],
                                             dropout=self.config['model']['dropout'],
                                             n_classes=self.config['data']['num_classes'],
                                             top_k=self.config['model']['topk'])           
        else:
            raise ValueError('Model not supported')
        return model
    
    def _assertions(self):
        if self.config['phase'] == 'train':
            assert self.model is not None, 'Model not provided'
        assert self.config, 'Config empty'

    def forward(self, batch):
        x, y = batch['img'], batch['label']
        if self.config['model']['attention'] == 'clam':
            return self.model(x, y, instance_eval=True, return_features=True, attention_only=False)
        elif self.config['model']['attention'] == 'dgrmil':
            if y.item() == 0:
                return self.model(x.unsqueeze(0), bag_mode='normal')
            else:
                return self.model(x.unsqueeze(0), bag_mode='abnormal')
        elif self.config['model']['attention'].startswith('bayesmil'):
            validation = not self.training
            # if self.config['logging']['model_version'] in ['v2', 'v3', 'v4', 'v5', 'v6']:   # v2 is apcrf
            coords = batch['coords']
            w, h = batch['w'], batch['h']
            return self.model(h=x, coords=coords, height=h[0], width=w[0], slide_label=y, validation=validation)
            # elif self.config['logging']['model_version']=='v1': # v1 is sdpr or enc
            #     return self.model(h=x, slide_label=y, validation=validation)
            # elif self.config['logging']['model_version']=='v0': # v0 is vis
            #     return self.model(h=x, validation=validation)
            # else:
            #     raise ValueError('BayesMIL Model not supported')
        elif self.config['model']['attention'] in ['abmil', 'transmil']:
            return self.model(x)
        else:
            raise ValueError('Model not supported')

    def on_train_start(self) -> None:
        # Print the model structure
        if self.print_model: print(self)

    def training_step(self, batch, batch_idx):
        # Data
        y = batch['label']
        # Model forward
        raw_out = self.forward(batch)
        if self.config['model']['attention'] in ['dgrmil']:
            Y_logits, A = raw_out['cls'], raw_out['A']
            Y_prob = F.softmax(Y_logits, dim=-1)
        elif self.config['model']['attention'] in ['bayesmil-spvis']:
            Y_logits, Y_prob, A = raw_out['top_instance'], raw_out['Y_prob'], raw_out['A']
        else:
            Y_prob, A = raw_out['Y_prob'], raw_out['A']
        
        # Compute loss
        loss_dict = self.compute_loss(raw_out, y)
        loss, bag_loss, instances_loss = (loss_dict['total_loss'], 
                                          loss_dict['bag_loss'], 
                                          loss_dict['instance_loss'])

        # Metrics
        self.train_accuracy.update(Y_prob.argmax(dim=-1), y)
        self.train_ece.update(Y_prob[:, 1].cpu(), y.cpu()) if self.num_classes == 2 else self.train_ece.update(Y_prob.cpu(), y.cpu())

        # Predictions and targets
        self.train_preds.append(Y_prob.clone().detach())
        self.train_targets.append(y.clone().detach())

        # Losses
        self.train_bag_loss.append(bag_loss)
        if instances_loss is not None:
            self.train_instance_loss.append(instances_loss)

        if self.config['model']['attention'] == 'dgrmil':
            wandb.log({'train/sim_loss': loss_dict['sim_loss']})
            wandb.log({'train/div_loss': loss_dict['div_loss']})

        return loss
    
    def on_train_epoch_end(self):
        # Compute the average train loss
        avg_bag_loss = torch.stack(self.train_bag_loss).mean()
        if self.train_instance_loss:
            avg_instance_loss = torch.stack(self.train_instance_loss).mean()
            avg_loss = avg_bag_loss + avg_instance_loss
        else:
            avg_instance_loss = torch.Tensor([0])
            avg_loss = avg_bag_loss

        y_true = torch.cat(self.train_targets, dim=0).cpu()
        y_pred = torch.stack(self.train_preds, dim=0).cpu().argmax(dim=-1)

        # losses as strings
        losses = {'bag_loss_fn': self.config['model']['bag_loss_fn'], 'instance_loss_fn': self.config['model']['instance_loss_fn']}
        
        wandb.log({
            'Epoch': self.current_epoch,
            'train/loss':{
                        f'train/bag_loss/{losses["bag_loss_fn"]}': avg_bag_loss.item(),
                        f'train/instance_loss/{losses["instance_loss_fn"]}': avg_instance_loss.item()
                },
            'train/avg_loss': avg_loss.item(),
            'train/accuracy': self.train_accuracy.compute().item(),
            'train/balanced_accuracy': balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
            'train/ece': self.train_ece.compute().item(),
        })

        self.train_accuracy.reset()
        self.train_bag_loss.clear()
        self.train_instance_loss.clear()
        self.train_ece.reset()
     
    def validation_step(self, batch, batch_idx):
        # Data
        x, y = batch['img'], batch['label']

        # Model forward
        raw_out = self.forward(batch)
        if self.config['model']['attention'] in ['dgrmil']:
            Y_logits, A = raw_out['cls'], raw_out['A']
            Y_prob = F.softmax(Y_logits, dim=-1)
        elif self.config['model']['attention'] in ['bayesmil-spvis']:
            Y_logits, Y_prob, A = raw_out['top_instance'], raw_out['Y_prob'], raw_out['A']
        else:
            Y_prob, A = raw_out['Y_prob'], raw_out['A']

        # Compute loss
        loss_dict = self.compute_loss(raw_out, y)
        loss, bag_loss, instances_loss = (loss_dict['total_loss'], 
                                          loss_dict['bag_loss'], 
                                          loss_dict['instance_loss'])
        
        self.val_bag_loss.append(bag_loss)
        if instances_loss is not None:
            self.val_instance_loss.append(instances_loss)

        # Metrics
        self.val_accuracy.update(Y_prob.argmax(dim=-1), y)
        self.val_auc.update(Y_prob[:, 1].cpu(), y.cpu()) if self.num_classes ==2 else self.val_auc.update(Y_prob.cpu(), y.cpu())
        self.val_cohenkappa.update(Y_prob[:, -1], y) if self.num_classes == 2 else self.val_cohenkappa.update(Y_prob, y)
        self.val_ece.update(Y_prob[:, 1].cpu(), y.cpu()) if self.num_classes == 2 else self.val_ece.update(Y_prob.cpu(), y.cpu())
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))

        if self.config['model']['attention'] == 'dgrmil':
            wandb.log({'val/sim_loss': loss_dict['sim_loss']})
            wandb.log({'val/div_loss': loss_dict['div_loss']})


        self.val_preds.append(Y_prob.clone().detach())
        self.val_targets.append(y.clone().detach())

        return loss

    def on_validation_epoch_end(self):
        avg_bag_loss = torch.stack(self.val_bag_loss).mean()
        if self.val_instance_loss:
            avg_instance_loss = torch.stack(self.val_instance_loss).mean()
            avg_loss = avg_bag_loss + avg_instance_loss
        else:
            avg_instance_loss = torch.Tensor([0])
            avg_loss = avg_bag_loss

        y_true = torch.cat(self.val_targets, dim=0).cpu()
        y_pred = torch.stack(self.val_preds, dim=0).cpu().argmax(dim=-1)
        
        # losses as strings
        losses = {'bag_loss_fn': self.config['model']['bag_loss_fn'], 'instance_loss_fn': self.config['model']['instance_loss_fn']}

        # Logs
        self.log('val/balanced_accuracy', balanced_accuracy_score(y_true=y_true, y_pred=y_pred))
        wandb.log({'Epoch':self.current_epoch,
                    'val/loss':{
                        f'val/bag_loss/{losses["bag_loss_fn"]}': avg_bag_loss.item(),
                        f'val/instance_loss/{losses["instance_loss_fn"]}': avg_instance_loss.item()
                    }, 
                    'val/avg_loss': avg_loss.item(),
                    'val/accuracy': self.val_accuracy.compute().item(),
                    'val/auc': self.val_auc.compute().item(),
                    'val/cohenskappa': self.val_cohenkappa.compute().item(),
                    'val/ece': self.val_ece.compute().item(),
                })
        
        # Clear stuff
        self.val_bag_loss.clear()
        self.val_instance_loss.clear()
        self.val_accuracy.reset()
        self.val_auc.reset()
        self.val_cohenkappa.reset()
        self.val_ece.reset()

    def on_train_end(self):
        # Convert predictions and targets to tensors
        preds_train = torch.stack(self.train_preds, dim=0).cpu()
        targets_train = torch.stack(self.train_targets, dim=0).cpu()
        preds_val = torch.stack(self.val_preds, dim=0).cpu()
        targets_val = torch.stack(self.val_targets, dim=0).cpu()

        # Clear the lists
        self.train_preds.clear()
        self.train_targets.clear()
        self.val_preds.clear()
        self.val_targets.clear()

        pass

    def test_step(self, batch, batch_idx):
        # Data again obvs
        y = batch['label']

        raw_out = self.forward(batch)
        if self.config['model']['attention'] in ['dgrmil']:
            Y_logits, A = raw_out['cls'], raw_out['A']
            Y_prob = F.softmax(Y_logits, dim=-1)
        elif self.config['model']['attention'] in ['bayesmil-spvis']:
            Y_logits, Y_prob, A = raw_out['top_instance'], raw_out['Y_prob'], raw_out['A']
        else:
            Y_prob, A = raw_out['Y_prob'], raw_out['A']

        # Metrics
        self.test_accuracy.update(Y_prob[:, 1], y) if self.num_classes == 2 else self.test_accuracy.update(Y_prob, y)
        self.test_auc.update(Y_prob[:, 1], y) if self.num_classes == 2 else self.test_auc.update(Y_prob, y)
        self.test_cohenkappa.update(Y_prob[:, 1], y) if self.num_classes == 2 else self.test_cohenkappa.update(Y_prob, y)
        self.test_confusion_matrix.update(Y_prob[:, 1], y) if self.num_classes == 2 else self.test_confusion_matrix.update(Y_prob, y)
        self.test_ece.update(Y_prob[:, 1], y) if self.num_classes == 2 else self.test_ece.update(Y_prob, y)

        self.test_preds.append(Y_prob)
        self.test_targets.append(y)
        pass

    def on_test_epoch_end(self):
        wandb.log({'test/accuracy': self.test_accuracy.compute().item(),
                   'test/balanced_accuracy': balanced_accuracy_score(y_true=torch.cat(self.test_targets, dim=0).cpu(), 
                                                                     y_pred=torch.stack(self.test_preds, dim=0).cpu().argmax(dim=-1)),
                  'test/auc': self.test_auc.compute().item(),
                  'test/cohenkappa': self.test_cohenkappa.compute().item(), 
                  'test/ece': self.test_ece.compute().item()
                  })
        wandb.log({'test/confusion_matrix': wandb.Image(self.test_confusion_matrix.plot()[0]),
                   'test/ece_curve': wandb.Image(self.test_ece.plot()[0]),
                   })

        self.test_accuracy.reset()
        self.test_auc.reset()
        self.test_cohenkappa.reset()
        self.test_confusion_matrix.reset()
        self.test_ece.reset()
        pass

    def compute_loss(self, raw_out: dict={}, y:int=-1):
        assert raw_out, 'No raw output provided'
        assert y >= 0, 'Invalid class label, should be >= 0'

        if self.config['model']['attention'] in ['clam']:
            return self._compute_loss_clam(raw_out, y)
        elif self.config['model']['attention'] in ['transmil']:
            return self._compute_loss_transmil(raw_out, y)
        elif self.config['model']['attention'] in ['abmil']:
            return self._compute_loss_abmil(raw_out, y)
        elif self.config['model']['attention'] in ['dgrmil']:
            return self._compute_loss_dgrmil(raw_out, y)
        elif self.config['model']['attention'] in ['bayesmil-spvis']:
            return self._compute_loss_bayesmil(raw_out, y)
        else:
            raise ValueError('Model not supported')

    def _compute_loss_clam(self, raw_out: dict={}, y:int=-1):
        assert raw_out['results_dict'], 'No results dict in raw output'

        # Define loss computation
        instance_loss = raw_out['results_dict']['instance_loss']
        if self.config['model']['bag_loss_fn'] == 'ce':
            bag_loss = CrossEntropyLoss()(raw_out['Y_prob'], y)
        elif self.config['model']['bag_loss_fn'] == 'svm':
            bag_loss = SmoothTop1SVM(n_classes=self.num_classes)(raw_out['logits'], y)
        else:
            raise ValueError('Bag loss function not supported, try ce or svm')

        bag_weight = self.config['model']['bag_weight']
        total_loss = bag_weight * bag_loss + (1-bag_weight) * instance_loss
        return {'total_loss': total_loss, 'instance_loss': instance_loss, 'bag_loss': bag_loss}
    
    def _compute_loss_transmil(self, raw_out: dict={}, y:int=-1):
        assert self.config['model']['bag_loss_fn'] in ['ce', 'svm'], 'Bag loss function not supported, try ce or svm'

        # Define loss computation
        instance_loss = None
        if self.config['model']['bag_loss_fn'] == 'ce':
            bag_loss = CrossEntropyLoss()(raw_out['Y_prob'], y)
        elif self.config['model']['bag_loss_fn'] == 'svm':
            bag_loss = SmoothTop1SVM(n_classes=self.num_classes)(raw_out['logits'], y)
        else:
            raise ValueError('Bag loss function not supported, try ce or svm')

        if instance_loss is not None:
            bag_weight = self.config['model']['bag_weight']
            total_loss = bag_weight * bag_loss + (1-bag_weight) * instance_loss
        else:
            total_loss = bag_loss
        
        return {'total_loss': total_loss, 'instance_loss': instance_loss, 'bag_loss': bag_loss}

    def _compute_loss_abmil(self, raw_out: dict={}, y:int=-1):
        assert self.config['model']['bag_loss_fn'] in ['ce', 'svm', 'bcewlogits'], 'Bag loss function not supported, try ce, svm or bcewlogits'

        # Define loss computation
        if self.config['model']['bag_loss_fn'] == 'ce':
            bag_loss = CrossEntropyLoss()(raw_out['Y_prob'], y)
        elif self.config['model']['bag_loss_fn'] == 'svm':
            bag_loss = SmoothTop1SVM(n_classes=self.num_classes)(raw_out['Y_logits'], y)
        elif self.config['model']['bag_loss_fn'] == 'bcewlogits':
            bag_loss = nn.BCEWithLogitsLoss()(raw_out['Y_logits'], y)
        else:
            raise ValueError('Bag loss function not supported, try ce, svm or bcewlogits')

        return {'total_loss': bag_loss, 'instance_loss': None, 'bag_loss': bag_loss}

    def _compute_loss_dgrmil(self, raw_out: dict={}, y:int=-1):
        epoch = self.current_epoch
        div_loss, sim_loss = 0, 0
        if epoch < self.config['training']['warmup_epochs']:
            if self.num_classes <=2:
                bag_loss = nn.BCEWithLogitsLoss()(raw_out['cls'][:,1], y.float())
            else:
                bag_loss = CrossEntropyLoss()(raw_out['cls'], y)
        else:
            # BCE
            if self.num_classes <= 2:
                bce = nn.BCEWithLogitsLoss()(raw_out['cls'][:, 1], y.float())
            else:
                bce = CrossEntropyLoss()(raw_out['cls'], y)
            # DIV
            lesion_norm = F.normalize(raw_out['lesion'].squeeze(0))
            div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1.e-10*torch.eye(lesion_norm.size(0)).cuda())
            # Similarity
            sim_loss = tripleloss(raw_out['lesion'], raw_out['p_center'], raw_out['nc_center'])
            bag_loss = bce + 0.1*div_loss + 0.1*sim_loss
        return {'total_loss': bag_loss, 'instance_loss': None, 'bag_loss': bag_loss, 'div_loss': div_loss, 'sim_loss': sim_loss}

    def _compute_loss_bayesmil(self, raw_out: dict={}, y:int=-1):
        logits = raw_out['top_instance']
        bayes_args = [get_ard_reg_vdo, 1.e-8, 1.e-12]

        if self.config['model']['bag_loss_fn'] == 'svm':
            bag_loss = SmoothTop1SVM(n_classes=self.num_classes)(logits, y)
        else:
            bag_loss = nn.CrossEntropyLoss()(logits, y)

        kl_model = bayes_args[0](self.model)
        validation = not self.training
        if validation:
            return {'total_loss': bag_loss, 'instance_loss': None, 'bag_loss': bag_loss}
        else:
            kl_data = 0.
            if 'spvis' in self.config['model']['attention']:
                kl_data = raw_out['kl_div'].reshape(-1).mean()
                total_loss = bag_loss + bayes_args[1] * kl_model + bayes_args[2] * kl_data
            elif 'enc' in self.config['model']['attention']:
                kl_data = raw_out['kl_div'].mean()
                total_loss = bag_loss + bayes_args[1] * kl_model + bayes_args[2] * kl_data
            elif 'vis' in self.config['model']['attention']:
                total_loss = bag_loss + bayes_args[1] * kl_model
            else:
                raise ValueError('BayesMIL Model not supported')
            
            return {'total_loss': total_loss, 'instance_loss': bayes_args[2]*kl_data, 'bag_loss': bag_loss}

    def configure_optimizers(self):
        # Define optimizer
        # Task for later: fix this method to utilize the optim_factory, get_optim and create_optim methods from MyOptimizer
        assert self.config['training']['optimizer'] in ['adam', 'adamw', 'sgd', 'lookahead_radam'], 'Optimizer not supported, must be adam, sgd or lookahead_radam'
        learning_rate = self.config['training']['learning_rate']
        if self.config['training']['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=self.config['training']['reg'])
        elif self.config['training']['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=self.config['training']['reg'])
        elif self.config['training']['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, weight_decay=self.config['training']['reg'], 
                                        momentum=self.config['training']['momentum'])
        elif self.config['training']['optimizer'] == 'lookahead_radam':
            opt_args = {'opt':'lookahead_radam', 'lr':learning_rate, 
                        'opt_eps':self.config['training']['opt_eps'], 
                        'opt_betas':self.config['training']['opt_betas'], 
                        'momentum':self.config['training']['momentum'],
                        'weight_decay':self.config['training']['reg']}
            opt_args_obj = DictToAttr(**opt_args)
            optimizer = create_optimizer(opt_args_obj, self.model)
        else:
            raise ValueError('Optimizer not supported')
        
        # Configure scheduler
        def lr_lambda(epoch):
            decay_after = self.config['training']['lr_decay_after_epoch']
            stop_decay_after = self.config['training']['stop_decay_after_epoch']
            lr_decay_value_stop = self.config['training']['stop_decay_lr_value']
            decay_factor = self.config['training']['lr_decay_factor']
            current_lr = optimizer.param_groups[0]['lr']

            if epoch < decay_after:
                return 1.0
            
            if epoch > stop_decay_after or current_lr <= lr_decay_value_stop:
                if self.first_low_lr_epoch is None:
                    self.first_low_lr_epoch = epoch

                return decay_factor ** (self.first_low_lr_epoch - decay_after + 1)
            
            return decay_factor ** (epoch - decay_after + 1)
        
        if self.config['training']['scheduler'] == 'lambda':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif self.config['training']['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['training']['max_epochs'],
                                                                   eta_min=self.config['training']['min_lr'])
        elif self.config['training']['scheduler'] == 'linearcosine':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.config['training']['warmup_epochs'],
                                                      max_epochs=self.config['training']['max_epochs'], eta_min=self.config['training']['min_lr'],
                                                      warmup_start_lr=self.config['training']['warmup_lr'])
        lr_scheduler_config = {'scheduler': scheduler, 'interval': self.config['training']['lr_logging_interval'],
                               'frequency': self.config['training']['lr_logging_frequency']}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}


class LitTSMIL(pl.LightningModule):
    def __init__(self, config: Dict = {}):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["model"])
        self.print_model = config["model"]["print_model"]

        self.num_classes = config["data"]["num_classes"]
        self._init_metrics()

        # -------------------------
        #  Build TSMIL model
        # -------------------------
        self.model = TSMIL(
            input_dim=config['model']['embed_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            head_dim=config['model']['head_dim'],
            cluster_num=config['model']['cluster_num'],
            T_depth=config['model']['T_depth'],
            mlp_ratio=config['model']['mlp_ratio'],
            dropout=config['model']['dropout'],
            aggregator=config['model']['aggregator'],
            num_classes=config['data']['num_classes'],
        )

        # for logging
        self.train_preds, self.val_preds, self.test_preds = [], [], []
        self.train_targets, self.val_targets, self.test_targets = [], [], []

    # -------------------------------------------------------------------------
    # Metrics Setup
    # -------------------------------------------------------------------------
    def _init_metrics(self):
        average = 'weighted'
        task = 'binary' if self.num_classes == 2 else 'multiclass'
        metrics_device = self.device

        # ----- Train -----
        self.train_accuracy = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average=average).to(metrics_device)
        self.train_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').cpu()

        # ----- Validation -----
        self.val_accuracy = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average=average).to(metrics_device)
        self.val_auc = torchmetrics.AUROC(task=task, num_classes=self.num_classes).cpu()
        self.val_cohenkappa = torchmetrics.CohenKappa(task=task, num_classes=self.num_classes, weights='quadratic').cpu()
        self.val_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').cpu()

        # ----- Test -----
        self.test_accuracy = torchmetrics.Accuracy(task=task, num_classes=self.num_classes, average=average).to(metrics_device)
        self.test_auc = torchmetrics.AUROC(task=task, num_classes=self.num_classes).cpu()
        self.test_cohenkappa = torchmetrics.CohenKappa(task=task, num_classes=self.num_classes, weights='quadratic').cpu()
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task=task, num_classes=self.num_classes).cpu()
        self.test_ece = torchmetrics.CalibrationError(task=task, num_classes=self.num_classes, n_bins=10, norm='l1').cpu()


    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, batch):
        x = batch["img"].unsqueeze(0)        # [B, N, D]
        logits = self.model(x) # [B, num_classes]
        Y_prob = torch.softmax(logits, dim=-1)
        return {"Y_prob": Y_prob, "logits": logits}

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        y = batch["label"]
        out = self.forward(batch)

        loss = nn.CrossEntropyLoss()(out["logits"], y)

        # metrics
        self.train_accuracy.update(out["Y_prob"].argmax(dim=-1), y)
        self.train_ece.update(out["Y_prob"][:, 1].cpu() if self.num_classes == 2 else out["Y_prob"].cpu(), y.cpu())

        self.train_preds.append(out["Y_prob"].detach().cpu())
        self.train_targets.append(y.detach().cpu())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def on_train_epoch_end(self):
        y_true = torch.cat(self.train_targets)
        y_pred = torch.stack(self.train_preds).argmax(dim=-1)

        wandb.log({
            "Epoch": self.current_epoch,
            "train/accuracy": self.train_accuracy.compute().item(),
            "train/balanced_accuracy": balanced_accuracy_score(y_true.numpy(), y_pred.numpy()),
            "train/ece": self.train_ece.compute().item()
        })

        self.train_accuracy.reset()
        self.train_ece.reset()
        self.train_targets.clear()
        self.train_preds.clear()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        out = self.forward(batch)

        loss = nn.CrossEntropyLoss()(out["logits"], y)

        self.val_accuracy.update(out["Y_prob"].argmax(dim=-1), y)
        self.val_auc.update(out["Y_prob"][:, 1].cpu() if self.num_classes == 2 else out["Y_prob"].cpu(), y.cpu())
        self.val_ece.update(out["Y_prob"][:, 1].cpu() if self.num_classes == 2 else out["Y_prob"].cpu(), y.cpu())

        self.val_preds.append(out["Y_prob"].detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_targets)
        y_pred = torch.stack(self.val_preds).argmax(dim=-1)

        self.log("val/balanced_accuracy", balanced_accuracy_score(y_true.numpy(), y_pred.numpy()),
         on_epoch=True, prog_bar=True)

        wandb.log({
            "Epoch": self.current_epoch,
            "val/accuracy": self.val_accuracy.compute().item(),
            "val/auc": self.val_auc.compute().item(),
            "val/balanced_accuracy": balanced_accuracy_score(y_true.numpy(), y_pred.numpy()),
            "val/ece": self.val_ece.compute().item()
        })

        self.val_accuracy.reset()
        self.val_auc.reset()
        self.val_ece.reset()
        self.val_targets.clear()
        self.val_preds.clear()

    # -------------------------------------------------------------------------
    # Testing
    # -------------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        y = batch["label"]
        out = self.forward(batch)

        self.test_accuracy.update(out["Y_prob"].argmax(dim=-1), y)
        self.test_auc.update(out["Y_prob"][:, 1] if self.num_classes == 2 else out["Y_prob"], y)
        self.test_cohenkappa.update(out["Y_prob"][:, 1] if self.num_classes == 2 else out["Y_prob"], y)
        self.test_confusion_matrix.update(out["Y_prob"][:, 1] if self.num_classes == 2 else out["Y_prob"], y)
        self.test_ece.update(out["Y_prob"][:, 1] if self.num_classes == 2 else out["Y_prob"], y)

        self.test_preds.append(out["Y_prob"].detach().cpu())
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        y_true = torch.cat(self.test_targets)
        y_pred = torch.stack(self.test_preds).argmax(dim=-1)

        wandb.log({
            "test/accuracy": self.test_accuracy.compute().item(),
            "test/auc": self.test_auc.compute().item(),
            "test/balanced_accuracy": balanced_accuracy_score(y_true.numpy(), y_pred.numpy()),
            "test/cohenkappa": self.test_cohenkappa.compute().item(),
            "test/confusion_matrix": wandb.Image(self.test_confusion_matrix.plot()[0]),
            "test/ece": self.test_ece.compute().item(),
        })

        self.test_accuracy.reset()
        self.test_auc.reset()
        self.test_cohenkappa.reset()
        self.test_confusion_matrix.reset()
        self.test_ece.reset()
        self.test_targets.clear()
        self.test_preds.clear()

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        cfg = self.config['training']

        # --------------------------
        # Optimizer
        # --------------------------
        opt_name = cfg['optimizer'].lower()
        lr = cfg['learning_rate']
        wd = cfg['reg']

        if opt_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                weight_decay=wd,
                momentum=cfg['momentum']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        # --------------------------
        # Scheduler
        # --------------------------
        sched_name = cfg['scheduler'].lower()

        if sched_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg['max_epochs'],
                eta_min=cfg['min_lr']
            )

        elif sched_name == 'linearcosine':
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=cfg['warmup_epochs'],
                max_epochs=cfg['max_epochs'],
                eta_min=cfg['min_lr'],
                warmup_start_lr=cfg['warmup_lr']
            )

        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")

        # Lightning requires a dict
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }



# For DGRMIL
def tripleloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)

    triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y) ,margin=1)
    
    loss = triple_loss(golabal,p_center,nc_center)

    return loss