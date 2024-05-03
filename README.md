<h1 align="left">
    GraphMETRO: Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts
</h1>

This repo is an early release of the official implementation for GraphMETRO - in reseponse to the great interests on our work.

The training scripts and datasets require additional cleanup before public release; however, the current files should serve as a useful reference and provide a general overview of our methodology.

# Environment

```
conda create -n graphmetro python=3.9
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
pip install pandas matplotlib networkx yacs seaborn torchmetrics ogb==1.3.6 munch dive-into-graphs
```

## Reference 

```
@article{graphmetro,
  author       = {Shirley Wu and
                  Kaidi Cao and
                  Bruno Ribeiro and
                  James Zou and
                  Jure Leskovec},
  title        = {GraphMETRO: Mitigating Complex Distribution Shifts in GNNs via Mixture
                  of Aligned Experts},
  year         = {2023},
  eprinttype    = {arXiv},
  eprint       = {2312.04693}
}
```
