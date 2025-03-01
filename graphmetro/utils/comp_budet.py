import math

from graphmetro.config import cfg, set_cfg
from graphmetro.model_builder import create_model


def params_count(model):
    '''
    Computes the number of parameters.
    Args:
        model (nn.Module): PyTorch model
    '''
    return sum([p.numel() for p in model.parameters()])


def get_stats(dataset):
    model = create_model(dataset, static=True)
    return params_count(model)


def match_computation(dataset, stats_baseline, key=['gnn', 'dim_inner'], mode='sqrt'):
    '''Match computation budget by modifying cfg.gnn.dim_inner'''
    stats = get_stats(dataset)
    if stats != stats_baseline:
        # Phase 1: fast approximation
        while True:
            if mode == 'sqrt':
                scale = math.sqrt(stats_baseline / stats)
            elif mode == 'linear':
                scale = stats_baseline / stats
            step = int(round(cfg[key[0]][key[1]] * scale)) \
                - cfg[key[0]][key[1]]
            cfg[key[0]][key[1]] += step
            stats = get_stats(dataset)
            if abs(step) <= 1:
                break
        # Phase 2: fine tune
        flag_init = 1 if stats < stats_baseline else -1
        step = 1
        while True:
            cfg[key[0]][key[1]] += flag_init * step
            stats = get_stats(dataset)
            flag = 1 if stats < stats_baseline else -1
            if stats == stats_baseline:
                return stats
            if flag != flag_init:
                if not cfg.model.match_upper:  # stats is SMALLER
                    if flag < 0:
                        cfg[key[0]][key[1]] -= flag_init * step
                    return get_stats(dataset)
                else:
                    if flag > 0:
                        cfg[key[0]][key[1]] -= flag_init * step
                    return get_stats(dataset)
    return stats


def match_baseline_cfg(dataset, cfg_dict, verbose=True):
    '''
    Match the computational budget of a given baseline model. THe current
    configuration dictionary will be modifed and returned.
    Args:
        cfg_dict (dict): Current experiment's configuration
        verbose (str, optional): If printing matched paramter conunts
    '''
    stats = match_computation(dataset, cfg.model.num_params, key=['gnn', 'dim_inner'])
    if 'gnn' in cfg_dict:
        cfg_dict['gnn']['dim_inner'] = cfg.gnn.dim_inner
    else:
        cfg_dict['gnn'] = {'dim_inner', cfg.gnn.dim_inner}
    if verbose:
        print('Computational budget has matched: Baseline params {}, '
              'Current params {}  Current inner dim {}'.format(cfg.model.num_params, stats, cfg.gnn.dim_inner))
    return 