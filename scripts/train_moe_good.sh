# SELECT ONE CONFIG FROM THE FOLLOWING
# AND COMMENT OUT THE REST


NAME=GOODSST2
TASK=graph
METRIC=acc
DIM=300
EPOCH=20
MOE_EPOCH=20
CONV=ginconv
ACT=dummy
N_LAYERS=3
LR=1e-4
MOE_LR=1e-3
BS=32
SCHEDULER=none
POOL=mean
DECAY=0.0
DROPOUT=0.5
K=[2]
P=[0.5]


NAME=GOODTwitter
TASK=graph
METRIC=acc
DIM=300
EPOCH=100
MOE_EPOCH=100
CONV=ginconv
ACT=dummy
N_LAYERS=3
LR=1e-4
MOE_LR=1e-3
BS=32
SCHEDULER=none
POOL=mean
DECAY=0.0
DROPOUT=0.5
K=[2]
P=[0.5]

NAME=GOODTwitch
TASK=node
CONV=gcnconv
METRIC=auc
DIM=128
EPOCH=300
MOE_EPOCH=300
ACT=dummy
N_LAYERS=3
LR=1e-4
MOE_LR=1e-2
BS=32
POOL=none
DROPOUT=0.5
DECAY=0.0
SCHEDULER=none
K=[2]
P=[0.5]

NAME=GOODWebKB
TASK=node
METRIC=acc
CONV=gcnconv
DIM=300
EPOCH=100
MOE_EPOCH=100
ACT=dummy
LR=1e-4
MOE_LR=1e-2
BS=32
POOL=none
DROPOUT=0.5
N_LAYERS=3
DECAY=0.0
SCHEDULER=none
K=[2]
P=[0.5]


NUM=5
GOOD_SINGLE=noisy_node_feat/add_edge/drop_edge/drop_node/random_subgraph
GOOD_PAIRED=noisy_node_feat/add_edge/drop_edge/drop_node/random_subgraph/noisy_node_feat-add_edge/noisy_node_feat-drop_edge/noisy_node_feat-random_subgraph/noisy_node_feat-drop_node/add_edge-drop_node/add_edge-random_subgraph/drop_edge-drop_node/drop_edge-random_subgraph/drop_node-random_subgraph


DATA_PATH=data/
WARM_UP=0
MODE=moe
SEED=0
AUGMENT=False
HARD=False
SHARED=False
SEED=0

python scripts/train.py --cfg scripts/default.yaml --wandb_exp_name paired-$NUM-experts  --wandb_id dsp-team \
    train.use_edge_attr False      train.augment $AUGMENT                 dataset.name $NAME \
    dataset.metric $METRIC         dataset.dir $DATA_PATH                 dataset.task $TASK \
    train.mode $MODE               gnn.layer_type $CONV                   train.batch_size $BS \
    gnn.act $ACT                   gnn.dim_inner $DIM                     optim.standard_lr $LR \
    optim.moe_lr $MOE_LR           optim.warmup_epoch $WARM_UP            gnn.layers_mp $N_LAYERS \
    optim.max_epoch $EPOCH         optim.moe_epoch $MOE_EPOCH             out_dir results/ \
    train.moe_hard $HARD           train.moe_shared $SHARED               seed $SEED \
    optim.scheduler $SCHEDULER     model.graph_pooling $POOL              gnn.dropout $DROPOUT \
    optim.weight_decay $DECAY      shift.k $K          shift.p $P         dataset.shift_type covariate \
    shift.train_types $GOOD_PAIRED  


