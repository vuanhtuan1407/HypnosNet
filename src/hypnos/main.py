import os

from dotenv import load_dotenv

import torch
import yaml
from lightning.fabric import Fabric

from src.hypnos.dataset import get_dataset
from src.hypnos.logger import get_logger
from src.hypnos.model import HypnosNet
from src.hypnos.training import train_hypnos
from src.hypnos.utils import training_args

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    args = training_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    logger = get_logger(config['logs']['log_dir'], config['logs']['log_level'])
    try:
        load_dotenv(config['env_path'])
    except Exception as e:
        logger.error("File .env may not exist. Continue without loading. Error: ", e)
    os.makedirs(config['out_dir'], exist_ok=True)  # create out dir
    fabric = Fabric(
        accelerator=config['train']["accelerator"],
        devices=config['train']["devices"],
        precision='32-true'
    )
    fabric.seed_everything(config['train']["seed"])
    fabric.launch()
    model = HypnosNet()
    try:
        eeg_dts = get_dataset([f"{config['data']['processed_data_dir']}/data_{i}.pkl" for i in range(config['data']['num_files'])])
        train_hypnos(fabric, model, eeg_dts, logger, config)
    except Exception as e:
        logger.error('Pickle file not found. Error: ', e)

