
import torch
import torch.nn.functional as F

import wandb
import os.path as osp
from graphmetro.transform import transform_func
from graphmetro.train.metric import compute_metric
from graphmetro.train.loss import compute_loss
from graphmetro.config import cfg
import logging
import numpy as np


NUM_SAMPLES = 3 if cfg.dataset.task == 'graph' else 20


def train_epoch(model, train_loader, optimizer, scheduler, augment=None):
    
    model.train()
    loss_all = 0
    
    for data in train_loader:
        data.to(cfg.device)
        
        # model forward with graph augmentation 
        if augment:
            if isinstance(augment, str):
                shift = augment
            elif isinstance(augment, list): 
                shift = np.random.choice(augment, size=1)[0]
                
            x, edge_index, edge_attr, batch, info = transform_func(shift)(data.x,
                                                                          data.edge_index,
                                                                          data.edge_attr,
                                                                          data.batch
                                                                          )
            if x.size(0) < 2 or edge_index.size(-1) < 2: 
                continue # skip small graphs
            out = model(x, edge_index, edge_attr, batch)
            
        # model forward without graph augmentation 
        else:
            info = None
            out = model(data.x,
                        data.edge_index,
                        data.edge_attr,
                        data.batch)
            
        # compute loss
        if cfg.dataset.task == 'graph':
            loss = compute_loss(out, data.y)
            
        elif cfg.dataset.task == 'node':
            # compute mask based on train_mask and the mask from augmentation
            mask = train_loader.dataset.train_mask.to(cfg.device).long()
            if (not info is None) and ('node_mask' in list(info.keys())):
                node_mask = info['node_mask'].long()
                y_mask = (mask * node_mask).bool()
                node_mask[node_mask == 1] = torch.arange(int(node_mask.sum())).to(cfg.device)
                out_mask = node_mask[y_mask]
            else:
                out_mask, y_mask = mask.bool(), mask.bool()
            loss = compute_loss(out[out_mask], data.y[y_mask])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    scheduler.step()
    
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def test(model, loaders, test_shifts=[]):
    model.eval()
    result = {}
    
    for split, loader in loaders.items():
        if split == 'test':
            shifts = test_shifts
        else:
            shifts = ['id']
            
        for shift in shifts:
            loss, metric, cnt = 0, 0, 0
            all_y_pred, all_y_true = [], []
            
            # for ood shifts, we sample multiple graphs for testing
            num_repeats = 1 if shift == 'id' else NUM_SAMPLES
            for _ in range(num_repeats):
                for data in loader:
                    data = data.to(cfg.device)
                    x, edge_index, edge_attr, batch, info = transform_func(shift)(data.x,
                                                                                  data.edge_index,
                                                                                  data.edge_attr,
                                                                                  data.batch
                                                                                  )
                    out = model(x, edge_index, edge_attr, batch)
                    
                    # evaluation
                    if cfg.dataset.task == 'graph':
                        
                        loss += compute_loss(out, data.y) * data.num_graphs
                        if loader.dataset.n_classes > 1:
                            out = out.argmax(dim=1)
                        elif cfg.dataset.metric == 'auc':
                            out = out.sigmoid()
                        elif cfg.dataset.metric == 'acc':
                            out = (out > 0.5).view(-1)
                            
                        metric += out.eq(data.y).sum()
                        cnt += data.num_graphs
                        all_y_true.append(data.y.view(-1))
                        all_y_pred.append(out.view(-1))
                        
                    elif cfg.dataset.task == 'node':
                        mask = getattr(loader.dataset, split + '_mask').to(cfg.device).long()
                        if (not info is None) and ('node_mask' in list(info.keys())):
                            node_mask = info['node_mask'].long()
                            y_mask = (mask * node_mask).bool()
                            node_mask[node_mask == 1] = torch.arange(int(node_mask.sum())).to(cfg.device)
                            out_mask = node_mask[y_mask]
                        else:
                            out_mask, y_mask = mask.bool(), mask.bool()
                        if torch.sum(y_mask) == 0:
                            continue # skip empty graph
                        
                        y_pred = out[out_mask].argmax(dim=1)
                        y_true = data.y[y_mask].long()
                        loss += compute_loss(out[out_mask], y_true) * torch.sum(y_mask)
                        metric += compute_metric(y_pred, y_true) * torch.sum(y_mask)
                        cnt += torch.sum(y_mask).item()
            
            result[f'{split}_{shift}_loss'] = loss.item() / cnt
            if cfg.dataset.task == 'graph' and cfg.dataset.metric == 'auc':
                all_y_pred = torch.concat(all_y_pred)
                all_y_true = torch.concat(all_y_true)
                result[f'{split}_{shift}'] = compute_metric(all_y_pred, all_y_true) 
            else:
                result[f'{split}_{shift}'] = float(metric.item()) / cnt
                
    return result


def train(model, loaders, optimizer, scheduler, augment):
    cnt = 0
    test_shifts = cfg.shift.test_types.split('/')
    logger = logging.getLogger('main')
    
    for epoch in range(1, cfg.optim.max_epoch + 1):
        loss = train_epoch(model, loaders['train'], optimizer, scheduler, augment)
        lr = scheduler.get_last_lr()[0]
        res = test(model, loaders, ['id'])
        if epoch == 1: 
            best_result = res
            
        train_res, val_res, test_res = res['train_id'], res['val_id'], res['test_id']
        logger.info(f'Epoch: {epoch:03d}, LR: {lr:.5f},  Loss: {loss:.4f}, '
                    f'Train: {train_res:.4f}, Val: {val_res:.4f}, Test: {test_res:.4f}')
        
        if val_res < best_result['val_id']:
            cnt = cnt + 1
        else: 
            if len(cfg.shift.test_types) > 0:
                res.update(
                    test(model, {'test': loaders['test']}, test_shifts)
                )
                logger.info(', '.join([f"({s}): {res[f'test_{s}']:.4f}" for s in test_shifts]))
            torch.save(model.cpu(), osp.join(cfg.out_dir, 'best_standard_model.pt'))
            model.to(cfg.device)
            cnt, best_result = 0, res
            
        if cfg.use_wandb:
            res['lr'] = lr
            wandb.log(res)
        
    return best_result, res

