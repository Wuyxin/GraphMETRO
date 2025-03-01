import wandb


def setup_wandb(cfg, args):
    if args.wandb_exp_name is None:
        wandb_exp_name = f'{cfg.dataset.name}-{cfg.train.mode}-seed={cfg.seed}'
    else:
        wandb_exp_name = f'{cfg.dataset.name}-{cfg.train.mode}-{args.wandb_exp_name}-seed={cfg.seed}'
    wandb.init(
        entity=args.wandb_id,
        project=args.wandb_project,
        settings=wandb.Settings(start_method="thread"),
        name=wandb_exp_name,
        config=args
    )
    
    
    
    