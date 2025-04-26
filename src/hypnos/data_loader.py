from torch.utils.data import DataLoader

from src.hypnos.data_utils import split_train_val_test, split_train_val_test_random


def get_loader(dataset, config):
    # train_set, val_set, test_set, _, _, _ = split_train_val_test(dataset)
    train_set, val_set, test_set, _, _, _ = split_train_val_test_random(dataset)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader, test_loader
