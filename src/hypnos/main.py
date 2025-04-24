import torch
import yaml
from lightning.fabric import Fabric

from src.hypnos.data_loader import get_loader
from src.hypnos.dataset import get_dataset
from src.hypnos.logger import get_logger
from src.hypnos.model import HypnosNet
from src.hypnos.training import fit
from src.hypnos.utils import training_args

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = training_args()
    config = yaml.load(open("./config.yml", "r"), Loader=yaml.FullLoader)

    logger = get_logger(config['logs']['log_dir'], config['logs']['log_level'])

    fabric = Fabric(
        accelerator=config['train']["accelerator"],
        devices=config['train']["devices"]
    )
    fabric.seed_everything(config['train']["seed"])
    fabric.launch()

    model = HypnosNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']["lr"])
    eeg_dataset = get_dataset(['../../data/processed/K3_EEG3_11h.pkl'], config)
    train_loader, val_loader, test_loader = get_loader(eeg_dataset, config)

    # setup to fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fit(fabric, model, train_loader, val_loader, optimizer, logger, config)
