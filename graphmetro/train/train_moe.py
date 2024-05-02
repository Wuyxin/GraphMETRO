
import torch
import numpy as np
import copy
import wandb
import os.path as osp
         
from gelato.train.loss import compute_loss
from gelato.train.metric import compute_metric
from gelato.transform import transform_func, get_gt_targets
from gelato.config import cfg
import logging
import warnings

NUM_SAMPLES = 3 if cfg.dataset.task == 'graph' else 20


def train_epoch_moe(epoch, model, 
                    train_loader, optimizer, 
                    scheduler, train_shifts
                    ):
    model.train()
    gating_loss_all, dist_loss_all, acc_all, cnt = 0, 0, 0, 0
    
    for i, data in enumerate(train_loader):
        loss = 0
        indices = np.random.randint(len(train_shifts), size=3) 
        for idx in indices:   
            trans_data = copy.deepcopy(data)
            
            trans_data.x, trans_data.edge_index, trans_data.edge_attr, \
                trans_data.batch, info = transform_func(train_shifts[idx])(data.x,
                                                                           data.edge_index,
                                                                           data.edge_attr,
                                                                           data.batch
                                                                           )
            gt_targets = get_gt_targets(train_shifts[idx], model.trans_func_names).to(cfg.device)
            
            if (not info is None) and 'node_mask' in list(info.keys()) and \
                cfg.dataset.task == 'node':
                node_mask = info['node_mask']
            else:
                node_mask = None
                
            data.to(cfg.device)
            trans_data.to(cfg.device)
            length = data.num_graphs if cfg.dataset.task == 'graph' else len(trans_data.batch)
            try:
                new_zs, gating_loss, z_dist_loss, acc = model.losses(data, 
                                                                        trans_data,
                                                                        node_mask, gt_targets, 
                                                                        hard=cfg.train.moe_hard
                                                                        )
            except ValueError:
                warnings.warn('Skip bad transformations.')
                continue
            
            # compute classification loss
            if epoch >= cfg.optim.warmup_epoch:
                out = model.classifier(new_zs)
                    
                if cfg.dataset.task == 'graph':
                    clas_loss = compute_loss(out, data.y)
                    
                elif cfg.dataset.task == 'node':
                    # compute mask based on train_mask and the mask from augmentation
                    mask = train_loader.dataset.train_mask.to(cfg.device).long()
                    if node_mask is not None:
                        node_mask = node_mask.long().to(cfg.device)
                        y_mask = (mask * node_mask).bool()
                        node_mask[node_mask == 1] = torch.arange(int(node_mask.sum())).to(cfg.device)
                        out_mask = node_mask[y_mask]
                    else:
                        out_mask, y_mask = mask.bool(), mask.bool()
                    clas_loss = compute_loss(out[out_mask], data.y[y_mask])
                    
                if torch.isnan(clas_loss):
                    clas_loss = torch.zeros_like(gating_loss).to(cfg.device)
                print('gating_loss', gating_loss.item(), 'z_dist', z_dist_loss.item(), 'clas', clas_loss.item())
            else:
                z_dist_loss, clas_loss = 0, 0
                print('gating_loss', gating_loss.item())
            # ablatiaon 
            z_dist_loss = 0
            loss += gating_loss + z_dist_loss + clas_loss
            cnt += length
            acc_all += acc * length
            gating_loss_all += gating_loss * length
            dist_loss_all += z_dist_loss * length             

        loss = loss / len(indices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        
    scheduler.step()
    
    return gating_loss_all / cnt, dist_loss_all / cnt, acc_all / cnt


@torch.no_grad()
def test_moe(model, 
             loaders, 
             test_shifts=[]
             ):
    model.eval()
    result, gating_result = {}, {}
    for split, loader in loaders.items():
        
        if split == 'test':
            shifts = test_shifts
        else:
            shifts = ['id']
            
        for idx, shift in enumerate(shifts):
            
            loss, metric, cnt = 0, 0, 0
            gating_acc, gating_choice = [], []
            num_repeats = 1 if shift == 'id' else NUM_SAMPLES
            gt_targets = get_gt_targets(shift, model.trans_func_names).to(cfg.device)
            
            for _ in range(num_repeats):
                all_y_pred, all_y_true = [], []
                for data in loader:
                    data = data.to(cfg.device)
                    trans_data = copy.deepcopy(data)
                    # while True:
                    trans_data.x, trans_data.edge_index, trans_data.edge_attr, \
                        trans_data.batch, info = transform_func(shift)(data.x,
                                                                       data.edge_index,
                                                                       data.edge_attr,
                                                                       data.batch
                                                                       )
                    out, gating_out = model(trans_data, 
                                            hard=cfg.train.moe_hard, 
                                            return_gating_out=True
                                            )
                    
                    gt_target = gt_targets.repeat(gating_out.size(0), 1).to(cfg.device)
                    gating_acc.append(((gating_out.sigmoid() > 0.5).float() == gt_target).float().mean().item())
                    gating_choice.append(gating_out.sigmoid().detach())
                    
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
            result[f'{split}_{shift}_acc'] = np.mean(gating_acc)
            
            gating_result[f'{split}_{shift}'] = torch.concat(gating_choice, dim=0).mean(dim=0).tolist()
            
    return result, gating_result
    
    
def train_moe(model, loaders, optimizer, scheduler):
    cnt = 0
    test_shifts = cfg.shift.test_types.split('/')
    train_shifts = cfg.shift.train_types.split('/') if len(cfg.shift.train_types) else model.trans_func_names
    logger = logging.getLogger('main')
    
    for epoch in range(1, cfg.optim.moe_epoch + cfg.optim.warmup_epoch + 1):
        gating_loss, dist_loss, acc = train_epoch_moe(epoch, 
                                                      model, 
                                                      loaders['train'], 
                                                      optimizer, 
                                                      scheduler, 
                                                      train_shifts
                                                      )
        lr = scheduler.get_last_lr()[0]
        res, gating_result = test_moe(model, loaders, test_shifts=['id'])
        
        if epoch == 1: 
            if len(cfg.shift.test_types) > 0:
                res_1, gating_result_1 = test_moe(model, 
                                                  loaders={'test': loaders['test']}, 
                                                  test_shifts=test_shifts
                                                  )
                res.update(res_1)
                gating_result.update(gating_result_1)
            best_result = res
            torch.save(model.cpu(), osp.join(cfg.out_dir, 'best_moe_model.pt'))
            model.to(cfg.device)
        
        train_res, val_res, test_id_res, freq = res['train_id'], res['val_id'], res['test_id'], gating_result[f'test_id']
        logger.info(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Gating loss: {gating_loss:.4f}, Gating ACC {acc:.4f}, Dist loss: {dist_loss:.4f}')
        logger.info(f'Train: {train_res:.4f}, Val: {val_res:.4f}, Test: {test_id_res:.4f}, Choice frequency: {freq}')
        
        if val_res <= best_result['val_id']:
            cnt = cnt + 1
        else: 
            if len(cfg.shift.test_types) > 0:
                res_2, gating_result_2 = test_moe(model, 
                                                    loaders={'test': loaders['test']}, 
                                                    test_shifts=test_shifts
                                                    )
                res.update(res_2)
                gating_result.update(gating_result_2)
                for s in test_shifts: 
                    logger.info(f"({s}): {res[f'test_{s}']:.4f} [acc={res[f'test_{s}_acc']:.4f}]")
                    logger.info(f"       choice frequency: {gating_result[f'test_id']}")
                    
            cnt, best_result = 0, res
            torch.save(model.cpu(), osp.join(cfg.out_dir, 'best_moe_model.pt'))
            model.to(cfg.device)
            
        if cfg.use_wandb:
            res['lr'] = lr
            wandb.log(res)
        
    return best_result, res