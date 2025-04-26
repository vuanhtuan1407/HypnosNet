import os
from pathlib import Path

import torch
from tqdm import tqdm
import wandb

from src.hypnos.data_loader import get_loader
from src.hypnos.train_utils import cal_kl_mse_cos_entropy_loss


def fit(fabric, model, train_loader, val_loader, optimizer, logger, config):
    model.train()
    for epoch in range(config['epochs']):
        logger.info(f'Epoch {epoch + 1}/{config["epochs"]}')
        total_loss = total_samples = 0
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training',
            disable=config['disable_tqdm']
        )
        for batch_idx, batch in train_tqdm:
            optimizer.zero_grad()
            sns, lbs_phs, dur, _ = batch
            z, lbs_phs_hat = model(sns)
            loss = cal_kl_mse_cos_entropy_loss(lbs_phs_hat, lbs_phs, config['kl_t'])
            fabric.backward(loss)

            # Stuck-Survival Training (Meta AI 2022)
            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * config.get('sst_noise', 1e-4)
                    # logger.debug(f'Gradient norm after SST of {n}: {p.grad.norm().item():.4f}')

            optimizer.step()
            total_loss += loss.item()
            total_samples += lbs_phs.shape[0]
            train_tqdm.set_postfix(loss=f'{total_loss / total_samples:.4f}')
        logger.info(f'Epoch {epoch + 1}/{config["epochs"]} - Train Loss: {total_loss / total_samples:.4f}')

        model.eval()
        val_loss = val_samples = 0
        with torch.no_grad():
            val_tqdm = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc='Validating',
                disable=config['disable_tqdm']
            )
            for batch_idx, batch in val_tqdm:
                sns, lbs_phs, dur, _ = batch
                z, lbs_phs_hat = model(sns)
                loss = torch.nn.functional.kl_div(torch.softmax(lbs_phs_hat, dim=1), lbs_phs, reduction='batchmean')

                if batch_idx == 0:
                    logger.debug(f'delta_lbs_phs: {torch.sub(lbs_phs, torch.softmax(lbs_phs_hat, dim=1))}')

                val_loss += loss.item()
                val_samples += lbs_phs.shape[0]
                val_tqdm.set_postfix(val_loss=f'{val_loss / val_samples:.4f}')
        logger.info(f'Epoch {epoch + 1}/{config["epochs"]} - Val Loss: {val_loss / val_samples:.4f}')


def test():
    pass


def train_hypnos(fabric, model, dataset, logger, config):
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        mode='online',
        project="HypnosNet",
        dir=Path(config['logs']['log_dir']).parent.absolute(),
        config=config['train']
    )
    train_config = wandb.config
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])
    train_loader, val_loader, test_loader = get_loader(dataset, train_config)

    # setup to fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fit(fabric, model, train_loader, val_loader, optimizer, logger, train_config)

    wandb.finish()
