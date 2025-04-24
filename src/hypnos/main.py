import torch
import yaml

from src.hypnos.data_loader import get_loader
from src.hypnos.dataset import get_dataset
from src.hypnos.model import HypnosNet
from src.hypnos.training import fit

if __name__ == '__main__':
    config = yaml.load(open("./config.yml", "r"), Loader=yaml.FullLoader)
    model = HypnosNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    eeg_dataset = get_dataset(['../../data/processed/K3_EEG3_11h.pkl'], config)
    train_loader, val_loader, test_loader = get_loader(eeg_dataset, config)
    fit(model, train_loader, val_loader, optimizer, config)
