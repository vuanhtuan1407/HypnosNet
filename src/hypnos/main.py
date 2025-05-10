import os

from dotenv import load_dotenv

import torch
import yaml
from lightning.fabric import Fabric

from src.hypnos.dataset import get_dataset
from src.hypnos.logger import get_logger
from src.hypnos.model import HypnosNet
from src.hypnos.prepare_data import prepare_data
from src.hypnos.training import train_hypnos
from src.hypnos.utils import parse_args, parse_data_args

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

    # get data
    if not os.path.exists(f"{config['data']['processed_data_dir']}/metainfo.yaml"):
        prepare_data(config['data'], logger)
    metainfo = yaml.load(open(f"{config['data']['processed_data_dir']}/metainfo.yaml", "r"), Loader=yaml.FullLoader)
    train_dataset = get_dataset(metainfo['train_files'])
    test_dataset = get_dataset(metainfo['test_files'])

    # training
    os.makedirs(config['train']['out_dir'], exist_ok=True)  # create out dir
    fabric = Fabric(
        accelerator=config['train']["accelerator"],
        devices=config['train']["devices"],
        precision='32-true'
    )
    fabric.seed_everything(config['train']["seed"])
    fabric.launch()
    model = HypnosNet()
    train_hypnos(fabric, model, train_dataset, test_dataset, config, logger)
