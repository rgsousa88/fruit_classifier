import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from model import model_factory

import os
from datetime import datetime

#Focal Loss (helps to handle unbalaced classes)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_losses(loss_name, **kwargs):
    loss_name = loss_name.lower()
    
    supported_losses = {
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCELoss,
        'bce_with_logits': nn.BCEWithLogitsLoss,
        'nll': nn.NLLLoss,
        'focal': FocalLoss
    }
    
    if loss_name not in supported_losses:
        raise ValueError(f"Invalid Loss '{loss_name}"
                        f"Use: {list(supported_losses.keys())}")
    
    return supported_losses[loss_name](**kwargs)

def get_optmizer(optimizer_name, params, lr=0.001, **kwargs):
    optimizer_name = optimizer_name.lower()
    
    supported_optimizers = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'rmsprop': optim.RMSprop,
        'adagrad': optim.Adagrad,
        'adadelta': optim.Adadelta
    }
    
    if optimizer_name not in supported_optimizers:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'")
    
    return supported_optimizers[optimizer_name](params, lr=lr, **kwargs)

def get_scheduler(sched_name, optimizer, **kwargs):
    sched_name = sched_name.lower()
    supported_scheds = {
        'step': (lr_scheduler.StepLR, ['step_size', 'gamma']),
        'cosine': (lr_scheduler.CosineAnnealingLR, ['T_max', 'eta_min']),
        'multistep': (lr_scheduler.MultiStepLR, ['milestones', 'gamma']),
        'exponential': (lr_scheduler.ExponentialLR, ['gamma']),
        'cyclic': (lr_scheduler.CyclicLR, ['base_lr', 'max_lr', 'step_size_up']),
        'reduce_lr': (lr_scheduler.ReduceLROnPlateau, ['mode', 'factor', 'patience']),
        'one_cycle': (lr_scheduler.OneCycleLR, ['max_lr', 'total_steps']),
        'plateau': (lr_scheduler.ReduceLROnPlateau, ['mode', 'factor', 'patience'])
    }

    if sched_name not in supported_scheds:
        raise ValueError(f"Invalid Scheduler '{sched_name}'")
    
    scheduler_class, valid_params = supported_scheds[sched_name]
    
    # Filtra apenas os parâmetros válidos
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    # Log de parâmetros ignorados
    ignored_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
    if ignored_params:
        print(f"Warning: Ignoring parameters for {sched_name}: {ignored_params}")
    
    return scheduler_class(optimizer, **filtered_kwargs)

class TrainHelper:
    def __init__(self, config:dict, evalFunc = None):
        self.config = config
        self.device = self.set_device(use_gpu=config['use_gpu'])
        
        self.model = model_factory(model_name=config['model'], num_classes=config['num_classes'], petrained=config['pretrained'])
        self.model = self.model.to(self.device)
        self.optimizer = get_optmizer(optimizer_name=config['optmizer'], params=self.model.parameters(), lr=config['lr'])
        self.criterion = get_losses(loss_name=config['loss'])

        scheduler_config = config.get('scheduler_param', {})
        self.scheduler = get_scheduler(sched_name=config['scheduler'], optimizer=self.optimizer, **scheduler_config)
        
        self.evalFunc = evalFunc
        self.checkpoint_path = ".\\checkpoint"
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def set_device(self,use_gpu=False):
        device = torch.device('cpu')

        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
        
        return device


    def trainStep(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        pred = self.model(data)
        loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metric = None
        if self.evalFunc:
            with torch.no_grad():
                metric = self.evalFunc(pred, target)

        return loss.item(), metric
    
    def valStep(self, data, target):
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            pred = self.model(data)
            loss = self.criterion(pred, target)
        
        metric = None
        if self.evalFunc:
            metric = self.evalFunc(pred, target)

        return loss.item(), metric
    
    def saveModel(self, epoch, val_loss, val_acc):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{self.config['model']}_{timestamp}.pth"
        fullPath = os.path.join(self.checkpoint_path, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'timestamp': timestamp,
            'epoch': epoch if epoch is not None else '-',
            'val_loss': val_loss if val_loss is not None else '-',
            'val_acc': val_acc if val_acc is not None else '-'
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, fullPath)
        print(f"Checkpoint salvo: {fullPath}")

    def loadModel(self, checkpoint):
        if self.config['model'] in checkpoint:
            state_dict = torch.load(checkpoint)
            self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
