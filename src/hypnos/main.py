import torch
import yaml
from lightning.fabric import Fabric

from src.hypnos.data_loader import get_loader
from src.hypnos.dataset import get_dataset
from src.hypnos.model import HypnosNet
from src.hypnos.training import fit

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    config = yaml.load(open("./config.yml", "r"), Loader=yaml.FullLoader)

    fabric = Fabric(
        accelerator=config["accelerator"],
        devices=config["devices"]
    )
    fabric.seed_everything(config["seed"])
    fabric.launch()

    model = HypnosNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    eeg_dataset = get_dataset(['../../data/processed/K3_EEG3_11h.pkl'], config)
    train_loader, val_loader, test_loader = get_loader(eeg_dataset, config)

    # setup to fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fit(fabric, model, train_loader, val_loader, optimizer, config)
