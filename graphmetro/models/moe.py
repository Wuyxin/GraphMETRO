import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList, Linear

import copy
import numpy as np
from graphmetro.config import cfg
        
        
class MoEModel(nn.Module):
    def __init__(self, trans_func_dict, gating_model, expert_models, **kwargs):
        super(MoEModel, self).__init__()
        self.num_experts = len(trans_func_dict)
        self.trans_funcs = list(trans_func_dict.values())
        self.trans_func_names = list(trans_func_dict.keys())
        self.experts = expert_models
        self.gating_model = gating_model
        self.classifier = self.experts[0].classifier
        for i in range(self.num_experts):
            self.experts[i].classifier = None
            
    def _aggregate(self, data, weights_or_labels, hard):
        '''
        Aggregate the outputs of different experts to obtain the final embeddings
        '''
        embs = []
        for i in range(self.num_experts):
            emb = self.experts[i].encoder(data.x, 
                                          data.edge_index, 
                                          data.edge_attr, 
                                          data.batch).unsqueeze(dim=-1)
            embs.append(emb)
        embs = torch.concat(embs, dim=-1)
        length = data.x.size(0) if cfg.dataset.task == 'node' else len(torch.unique(data.batch))
        
        if isinstance(weights_or_labels, (int, np.int64)):
            weights_or_labels = torch.LongTensor([weights_or_labels for _ in range(length)])
            
        if len(weights_or_labels.size()) == 1:
            index = copy.deepcopy(weights_or_labels)
            weights = torch.zeros((length, embs[0].size(1))).to(cfg.device)
            weights[range(length), index] = 1.
        elif hard:
            mask = torch.zeros((length, embs[0].size(1))).to(cfg.device)
            mask[range(length), weights_or_labels.argmax(dim=1)] = 1
            mask = mask.bool()
            weights = torch.zeros_like(weights_or_labels)
            weights[mask] = weights_or_labels[mask]
        else:
            weights = weights_or_labels
            
        weights = weights.unsqueeze(dim=2)
        new_zs = torch.bmm(embs, weights).squeeze(dim=-1)
        
        return new_zs
    
    def get_expert_weights(self, trans_data, softmax=True):
        
        expert_weights = self.gating_model(trans_data.x, 
                                           trans_data.edge_index, 
                                           trans_data.edge_attr, 
                                           trans_data.batch
                                           )
        if softmax:
            expert_weights = expert_weights.softmax(dim=-1)
        return expert_weights
    
    
    def forward(self, trans_data, hard=True, return_gating_out=True):
        '''
        Model forward of the MoE model
        '''
        expert_weights = self.get_expert_weights(trans_data, softmax=True)
        indices = expert_weights.argmax(dim=-1).view(-1)
        new_z = self._aggregate(trans_data, expert_weights, hard)
        out = self.classifier(new_z)
        if return_gating_out:
            if hard:
                expert_weights = torch.zeros_like(expert_weights)
                expert_weights[range(expert_weights.size(0)), indices] = 1
            return out, expert_weights
        return out
    
    def losses(self, data, trans_data, node_mask, targets, hard=True):
        '''
        data:           original graph
        trans_data:     transformed graph
        node_mask:      node mask apply on data to obtain trans_data
        y:              the environment label of data and trans_data
        hard:           whether to use hard infer, if so, taking the output 
                        of the expert with largest weights
        '''
        expert_weights = self.get_expert_weights(trans_data, softmax=False)
        
        # compute the embedding based on expert weights
        target = targets.repeat(expert_weights.size(0), 1).to(cfg.device)
        print('Added pos_weight=5')
        gating_loss = F.binary_cross_entropy_with_logits(expert_weights, target, 
                                                         pos_weight=torch.Tensor([4.]).to(expert_weights.device)
                                                         )
        acc = ((expert_weights.sigmoid() > 0.5).float() == target).float().mean()
        
        # add detach
        use_softmax = True
        if use_softmax:
            new_zs = self._aggregate(trans_data, expert_weights.detach().softmax(dim=-1), hard)
        else:
            new_zs = self._aggregate(trans_data, expert_weights.detach().sigmoid(), hard)
        
        # gt_zs = self._aggregate(data, y, hard)
        # gt_zs_trans = self._aggregate(trans_data, y, hard)
        gt_zs_base = self._aggregate(data, 0, hard)
        
        # compute the embedding as the ground truth expert output
        length = data.num_graphs if cfg.dataset.task == 'graph' else len(trans_data.batch)
        
        if node_mask is not None:
            gt_zs_base = gt_zs_base[node_mask]
            # gt_zs = gt_zs[node_mask]
        
        z_dist =  torch.linalg.norm(new_zs - gt_zs_base, 'fro') / length # + torch.linalg.norm(gt_zs - gt_zs_trans, 'fro') / length 
        
        return new_zs, gating_loss, z_dist, acc
    