<h1 align="left">
    GraphMETRO: Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts (NeurIPS 2024)
</h1>


## Reference 

Please consider citing our paper:
```
@inproceedings{wu24graphmetro,
    author     = {Shirley Wu and
                  Kaidi Cao and
                  Bruno Ribeiro and
                  James Zou and
                  Jure Leskovec},
    title      = {GraphMETRO: Mitigating Complex Distribution Shifts in GNNs via Mixture of Aligned Experts},
    booktitle  = {NeurIPS},
    year       = {2024}
}
```

## Running the code
```
sh scripts/train_moe.sh
sh scripts/train_moe_good.sh
```
Please specify your own `wandb_id` in `scripts/*.sh` if `use_wandb` is set to True.


# Environment

```
conda create -n graphmetro python=3.9
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install pandas matplotlib networkx yacs seaborn torchmetrics ogb==1.3.6 munch dive-into-graphs
```

