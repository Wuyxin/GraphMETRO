# SELECT ONE CONFIG FROM THE FOLLOWING
# AND COMMENT OUT THE REST

NAME=TU_REDDIT-BINARY
TASK=graph
METRIC=acc
DIM=128
EPOCH=100
MOE_EPOCH=100
CONV=gatconv
ACT=prelu
N_LAYERS=2
LR=1e-3
MOE_LR=1e-3
BS=32
SCHEDULER=none
POOL=add
DECAY=0.0
DROPOUT=0.0
K=[2]
P=[0.5]


NAME=CiteSeer
TASK=node
CONV=gatconv
METRIC=acc
DIM=32
EPOCH=200
MOE_EPOCH=200
ACT=prelu
POOL=none
N_LAYERS=3
LR=1e-3
MOE_LR=1e-3
DROPOUT=0.0
DECAY=0.0
BS=32
SCHEDULER=none
K=[2]
P=[0.5]


NAME=TU_IMDB-MULTI
TASK=graph
METRIC=acc
DIM=128
EPOCH=200
MOE_EPOCH=200
CONV=gatconv
ACT=prelu
N_LAYERS=2
LR=1e-4
MOE_LR=1e-4
BS=32
SCHEDULER=none
POOL=add
DECAY=0.0
DROPOUT=0.0
K=[2]
P=[0.5]


NAME=DBLP
TASK=node
CONV=gatconv
METRIC=acc
DIM=64
EPOCH=100
MOE_EPOCH=100
ACT=prelu
N_LAYERS=3
LR=1e-3
MOE_LR=1e-3
DROPOUT=0.0
POOL=none
DECAY=0.0
BS=32
SCHEDULER=none
K=[2]
P=[0.5]


single=noisy_node_feat/add_edge/drop_edge/drop_node/random_subgraph
paired=noisy_node_feat/add_edge/drop_edge/drop_node/random_subgraph/noisy_node_feat-add_edge/noisy_node_feat-drop_edge/noisy_node_feat-random_subgraph/noisy_node_feat-drop_node/random_subgraph-add_edge/random_subgraph-drop_edge

WARM_UP=0
DATA_PATH=path/to/store_data
MODE=moe
SEED=0
AUGMENT=False
HARD=False
SHARED=False
python scripts/train.py --cfg scripts/default.yaml --use_wandb 1 --wandb_exp_name paired-shared \
 train.use_edge_attr False      train.augment $AUGMENT                 dataset.name $NAME \
 dataset.metric $METRIC         dataset.dir $DATA_PATH                 dataset.task $TASK \
 train.mode $MODE               gnn.layer_type $CONV                   train.batch_size $BS \
 gnn.act $ACT                   gnn.dim_inner $DIM                     optim.standard_lr $LR \
 optim.moe_lr $MOE_LR           optim.warmup_epoch $WARM_UP            gnn.layers_mp $N_LAYERS \
 optim.max_epoch $EPOCH         optim.moe_epoch $MOE_EPOCH             out_dir results/ \
 train.moe_hard $HARD           seed $SEED                             optim.scheduler $SCHEDULER \
 model.graph_pooling $POOL      gnn.dropout $DROPOUT                   optim.weight_decay $DECAY \
 shift.k $K                     shift.p $P                             dataset.shift_type covariate \
 shift.train_types $paired      shift.test_types $paired               train.moe_shared $SHARED
