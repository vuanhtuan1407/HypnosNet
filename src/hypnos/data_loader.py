import os

import joblib
from torch.utils.data import DataLoader, Subset

from src.hypnos.data_utils import split_train_val_test_random

indices_files = ['train_indices.pkl', 'val_indices.pkl', 'test_indices.pkl']


def get_loader(dataset, config):
    # train_set, val_set, test_set, _, _, _ = split_train_val_test(dataset)
    # train_set, val_set, test_set, _, _, _ = split_train_val_test_random(dataset)

    indices_paths = [os.path.join(config["processed_data_dir"], f) for f in indices_files]

    if not all(os.path.exists(path) for path in indices_paths):
        train_set, val_set, test_set, train_indices, val_indices, test_indices = split_train_val_test_random(dataset)

        for indices, path in zip([train_indices, val_indices, test_indices], indices_paths):
            joblib.dump(indices, path)
    else:
        train_indices = joblib.load(indices_paths[0])
        val_indices = joblib.load(indices_paths[1])
        test_indices = joblib.load(indices_paths[2])

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        test_set = Subset(dataset, test_indices)

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
