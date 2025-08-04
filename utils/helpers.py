import json
import torch
import logging
from pathlib import Path
import yaml
from argparse import Namespace
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def str2none(v):
    if v is None:
        return v
    if v.lower() == 'none':
        return None
    else:
        return v


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else 0


def count_untrainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad) if model is not None else 0



def build_optim(args, params):
    optimiser = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    return optimiser


def get_training_logger(checkpoint_dir: Path,
                        log_name: str = "E_DDPG_training_4X_Test4.log",
                        overwrite: bool = True) -> logging.Logger:
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_file = checkpoint_dir / log_name
    mode = "w" if overwrite else "a"

    logger = logging.getLogger(f"{__name__}.{log_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False        

    if logger.handlers:
        logger.handlers.clear()

    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(log_file, mode=mode)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def _to_namespace(obj):
    if isinstance(obj, dict):
        return Namespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_yaml(cfg_path: str | Path) -> Namespace:
    cfg_path = Path(cfg_path).expanduser()
    with cfg_path.open("r") as f:
        cfg_dict = yaml.safe_load(f)

    return _to_namespace(cfg_dict)
