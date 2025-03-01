import os
import os.path as osp
import numpy as np
import torch
from collections import OrderedDict
import argparse
import re
from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler

import sys
sys.path.append(".")
from graphmetro.transform import get_trans_func_dict
from graphmetro.model_builder import create_model
from graphmetro.datasets import get_datasets
from graphmetro.config import cfg, dump_cfg, load_cfg, set_out_dir
from graphmetro.utils.device import auto_select_device
from graphmetro.utils.comp_budet import match_baseline_cfg
from graphmetro.train.optimizer import create_optimizer, create_scheduler
from graphmetro.train.train_standard import train
from graphmetro.logger import Logger
from graphmetro.utils.seed import set_seed
from graphmetro.utils.wandb import setup_wandb
from graphmetro.transform import *
from graphmetro.models.layers import MLP, Wrapper

import warnings
warnings.simplefilter("ignore")
np.set_printoptions(precision=4)


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='graphmetro')

    parser.add_argument('--cfg',
                        dest='cfg_file',
                        type=str,
                        default=None,
                        help='The configuration file path.')
    parser.add_argument("--use_wandb",
                        type=int, default=1
                        )
    parser.add_argument("--model_path",
                        type=str, default=None,
                        help='The model path of the trained GNN'
                        )
    parser.add_argument("--wandb_id",
                        type=str, default='dsp-team', # <your team name here>
                        help='The wandb username for logging'
                        )
    parser.add_argument("--wandb_project",
                        type=str, default='graphmetro',
                        help='The wandb project name for logging'
                        )
    parser.add_argument("--wandb_exp_name",
                        type=str, default=None,
                        help='The wandb experiment name for logging'
                        )
    parser.add_argument('opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()


auto_select_device()
args = parse_args()
load_cfg(cfg, args)
set_seed(cfg.seed)
set_out_dir(cfg)
if args.use_wandb:
    setup_wandb(cfg, args)
torch.set_num_threads(cfg.num_threads)

os.makedirs(cfg.dataset.dir, exist_ok=True)
datasets = get_datasets(cfg.dataset.name, root=cfg.dataset.dir)

data = datasets['train'][0]
node_feat_dim = data.x.size(-1)
edge_feat_dim = 1 if (data.edge_attr is None or len(data.edge_attr.size()) == 1) \
                  else data.edge_attr.size(-1)
                  
logger = Logger.get_logger(name='main', fname=osp.join(cfg.out_dir, 'output.log'))
logger.info(f'Training data {data} \n data.x {data.x[:5]}')
batch_size = cfg.train.batch_size

loaders = {}
loaders['train'] = DataLoader(datasets['train'], 
                              batch_size=batch_size, 
                              shuffle=True
                              )
loaders['val'] = DataLoader(datasets['val'], 
                            batch_size=batch_size, 
                            shuffle=False
                            )
loaders['test'] = DataLoader(datasets['test'], 
                             batch_size=batch_size, 
                             shuffle=False
                             )

trans_func_names = ['id'] + list(set(re.split('-|/', cfg.shift.train_types)))
trans_func_dict = get_trans_func_dict(trans_func_names)
logger.info(f'Transform functions: {list(trans_func_dict.keys())}')
    

if cfg.train.mode == 'standard' or \
   (cfg.train.mode == 'moe_shared' and args.model_path is None):
    if args.model_path:
        model = torch.load(args.model_path).to(cfg.device)
        logger.info(f'Resume from {args.model_path}')
    else:
        model = create_model(datasets['train'])
        
    optimizer = create_optimizer(model.parameters(), lr=cfg.optim.standard_lr)
    scheduler = create_scheduler(optimizer)
    augment = list(trans_func_dict.keys()) if cfg.train.augment else None
    
    best_result, last_result = train(model,
                                     loaders,
                                     optimizer, 
                                     scheduler,
                                     augment
                                     )
    
    torch.save(model.cpu(), osp.join(cfg.out_dir, 'last_standard_model.pt'))

if cfg.train.mode == 'moe_shared':
    from graphmetro.models.moe_shared import MoEModel
    from graphmetro.train.train_moe_shared import train_moe
    
    gating_model = create_model(datasets['train'], dim_out=len(trans_func_dict))
    if args.model_path is None:
        args.model_path = osp.join(cfg.out_dir, 'best_standard_model.pt')
    model = MoEModel(trans_func_dict=trans_func_dict, 
                     gating_model=gating_model
                     ).to(cfg.device)
    optimizer = create_optimizer(model.parameters(), lr=cfg.optim.moe_lr)
    scheduler = create_scheduler(optimizer)
    gnn_model = torch.load(args.model_path).to(cfg.device)
    
    best_result, last_result = train_moe(model, 
                                         gnn_model, 
                                         loaders, 
                                         optimizer, 
                                         scheduler
                                         )
    torch.save(model.cpu(), osp.join(cfg.out_dir, 'last_moe_model.pt'))
    
if cfg.train.mode == 'moe':
    from graphmetro.models.moe import MoEModel
    from graphmetro.train.train_moe import train_moe
    gating_model = create_model(datasets['train'], dim_out=len(trans_func_dict))
    if args.model_path:
        base_model = torch.load(args.model_path)
        logger.info(f'Resume from {args.model_path}')
    else:
        base_model = create_model(datasets['train'], has_post_mp=True)
        
    if cfg.train.moe_shared:
        encoder = base_model.encoder
        detach = False if args.model_path is None else True
        if detach:
            for params in encoder.parameters():
                params.requires_grad = False
        expert_models = torch.nn.ModuleList([base_model] + 
            [Wrapper(encoder, MLP(dim_out=cfg.gnn.dim_inner, 
                                  skip_connection=True, 
                                  act=torch.nn.Sigmoid()))
             for _ in range(len(trans_func_dict) - 1)
             ])
    else:
        expert_models = torch.nn.ModuleList([base_model] + 
                                            [create_model(datasets['train'], has_post_mp=False) \
                                             for _ in range(len(trans_func_dict) - 1)])
    model = MoEModel(trans_func_dict=trans_func_dict, 
                     gating_model=gating_model,
                     expert_models=expert_models
                     ).to(cfg.device)
    optimizer = torch.optim.Adam([
        {'params': model.gating_model.parameters(), 'lr': cfg.optim.moe_lr},
        {'params': model.experts.parameters(), 'lr': cfg.optim.moe_lr},
        {'params': model.classifier.parameters(), 'lr': cfg.optim.standard_lr}
        ], weight_decay=cfg.optim.weight_decay
    )
    scheduler = create_scheduler(optimizer)
    
    best_result, last_result = train_moe(model, 
                                         loaders, 
                                         optimizer, 
                                         scheduler
                                         )
    torch.save(model.cpu(), osp.join(cfg.out_dir, 'last_moe_model.pt'))


# logs
logger.info('=' * 100)
logger.info('Best result' + f'{best_result}')
logger.info('Last result' + f'{last_result}')

import csv
if (not os.path.isfile(osp.join(cfg.out_dir, '..', 'best_result.csv'))):
    writeheader = True
else:
    writeheader = False
    
with open(osp.join(cfg.out_dir, '..', 'best_result.csv'), 'a+') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['exp_name', 'best_result'])
    if writeheader: writer.writeheader()
    writer.writerows([{'exp_name': cfg.out_dir.split('/')[-1], 'best_result': best_result}])
    
with open(osp.join(cfg.out_dir, '..', 'last_result.csv'), 'a+') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['exp_name', 'last_result'])
    if writeheader: writer.writeheader()
    writer.writerows([{'exp_name': cfg.out_dir.split('/')[-1], 'last_result': best_result}])