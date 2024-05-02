import torch.optim as optim
from graphmetro.config import cfg


def create_optimizer(params, lr=None):
    r"""Creates a config-driven optimizer."""
    params = filter(lambda p: p.requires_grad, params)
    if cfg.optim.optimizer == 'adam':
        optimizer = optim.Adam(params,
                               lr=cfg.optim.standard_lr if lr is None else lr,
                               weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'sgd':
        optimizer = optim.SGD(params,
                              lr=cfg.optim.standard_lr if lr is None else lr,
                              momentum=cfg.optim.momentum,
                              weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError('Optimizer {} not supported'.format(
            cfg.optim.optimizer))

    return optimizer


def create_scheduler(optimizer):
    r"""Creates a config-driven learning rate scheduler."""
    if cfg.optim.scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.optim.max_epoch + 1,
                                              gamma=0.1)
    elif cfg.optim.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.optim.max_epoch // 3,
                                              gamma=0.1)
    elif cfg.optim.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.optim.steps,
                                                   gamma=cfg.optim.lr_decay)
    elif cfg.optim.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.optim.max_epoch)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            cfg.optim.scheduler))
    return scheduler


