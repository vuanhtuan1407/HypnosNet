import os
from pathlib import Path

import torch
import wandb
import yaml
from dotenv import load_dotenv
from lightning.fabric import Fabric

from src.hypnos.dataset import get_dataset
from src.hypnos.logger import get_logger
from src.hypnos.params import MODEL_DATASET_MAP
from src.hypnos.prepare_data import prepare_data
from src.hypnos.training import train_model
from src.hypnos.utils import parse_args

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    logger = get_logger(config['logs']['log_dir'], config['logs']['log_level'])

    # load .env
    try:
        load_dotenv(config['train']['env_path'])
    except Exception as e:
        logger.error("File .env may not exist. Continue without loading. Error: ", e)

    # define wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        mode='disabled',
        project="HypnosNet",
        dir=Path(config['logs']['log_dir']).parent.absolute(),
        config=config['train'],
    )

    # define fabrics
    fabric = Fabric(
        accelerator=config['train']["accelerator"],
        devices=config['train']["devices"],
        precision='32-true',
    )
    fabric.seed_everything(config['train']["seed"])
    fabric.launch()

    model_name = config['train']['model_name']

    # get data
    if not os.path.exists(f"{config['data']['processed_data_dir']}/metainfo.yaml"):
        prepare_data(config['data'], postfix='all', logger=logger)
    metainfo = yaml.load(open(f"{config['data']['processed_data_dir']}/metainfo.yaml", "r"), Loader=yaml.FullLoader)

    train_dataset = get_dataset(metainfo['train_files'][MODEL_DATASET_MAP[model_name]])
    val_dataset = get_dataset(metainfo['val_files'])
    test_dataset = get_dataset(metainfo['test_files'])

    # training
    os.makedirs(config['train']['out_dir'], exist_ok=True)  # create out dir
    train_model(fabric, model_name, train_dataset, val_dataset, test_dataset, config, wandb, logger)

    wandb.finish()
