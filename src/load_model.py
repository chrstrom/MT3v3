import os
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from util.load_config_files import load_yaml_into_dotdict

from modules.models.mt3v3.mt3v3 import MOTT

def parse_input_args():
    # Load CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tp",
        "--task_params",
        help="filepath to configuration yaml file defining the task",
        required=True,
    )
    parser.add_argument(
        "-mp",
        "--model_params",
        help="filepath to configuration yaml file defining the model",
        required=True,
    )
    parser.add_argument(
        "--continue_training_from",
        help="filepath to folder of an experiment to continue training from",
    )
    args = parser.parse_args()

    # Load hyperparameters
    params = load_yaml_into_dotdict(args.task_params)
    params.update(load_yaml_into_dotdict(args.model_params))
    eval_params = load_yaml_into_dotdict(args.task_params)
    eval_params.update(load_yaml_into_dotdict(args.model_params))
    eval_params.recursive_update(load_yaml_into_dotdict("configs/eval/default.yaml"))

    # Generate 32-bit random seed, or use user-specified one
    if params.general.pytorch_and_numpy_seed is None:
        random_data = os.urandom(4)
        params.general.pytorch_and_numpy_seed = int.from_bytes(
            random_data, byteorder="big"
        )

    if params.training.device == "auto":
        params.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    if eval_params.training.device == "auto":
        eval_params.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Task configuration file: {args.task_params}")
    print(f"Model configuration file: {args.model_params}")
    print(f"Using seed: {params.general.pytorch_and_numpy_seed}")
    print(
        f"Training device: {params.training.device}, Eval device: {eval_params.training.device}, Model: {params.arch.type}"
    )

    return args, params, eval_params


if __name__ == '__main__':

    # Load CLI arguments
    args, params, eval_params = parse_input_args()

    # Generate 32-bit random seed, or use user-specified one
    if params.general.pytorch_and_numpy_seed is None:
        random_data = os.urandom(4)
        params.general.pytorch_and_numpy_seed = int.from_bytes(random_data, byteorder="big")
        
    seed = params.general.pytorch_and_numpy_seed
        
    print(f'Using seed: {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    model = MOTT(params)
    model.to(torch.device(params.training.device))
    