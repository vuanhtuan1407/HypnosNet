import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wandb

from src.hypnos.data_loader import get_loader
from src.hypnos.train_utils import cal_kl_mse_cos_entropy_loss


def fit(fabric, model, train_loader, val_loader, optimizer, logger, config):
    model.train()
    for epoch in range(config['epochs']):
        total_loss = total_samples = 0
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training Epoch {epoch + 1}/{config["epochs"]}',
            disable=config['disable_tqdm']
        )
        for batch_idx, batch in train_tqdm:
            optimizer.zero_grad()
            sns, lbs_phs, dur, _ = batch
            z, lbs_phs_hat = model(sns)
            kl_t = 1 + config['kl_e'] / (epoch + config['kl_e'])
            loss = cal_kl_mse_cos_entropy_loss(lbs_phs_hat, lbs_phs, kl_t)
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
        wandb.log({'train/loss': total_loss / total_samples}, step=epoch)

        model.eval()
        # val_loss = val_samples = 0
        best_metric = 1e6
        preds, truths = [], []
        with torch.no_grad():
            val_tqdm = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f'Validating Epoch {epoch + 1}/{config["epochs"]}',
                disable=config['disable_tqdm']
            )
            for batch_idx, batch in val_tqdm:
                sns, lbs_phs, dur, _ = batch
                z, lbs_phs_hat = model(sns)
                preds.append(lbs_phs_hat)
                truths.append(lbs_phs)

            preds = torch.cat(preds, dim=0)
            truths = torch.cat(truths, dim=0)
            log_preds = torch.nn.functional.log_softmax(preds, dim=1)
            preds = torch.nn.functional.softmax(log_preds, dim=1)
            kl_div = torch.nn.functional.kl_div(log_preds, truths, reduction='batchmean')
            cos_sim = torch.nn.functional.cosine_similarity(preds, truths, dim=1).mean()
            mse = torch.nn.functional.mse_loss(preds, truths)
            logger.info(
                f'Epoch {epoch + 1}/{config["epochs"]} - '
                f'Val KLDiv: {kl_div.item():.4f} - '
                f'Val CosSim: {cos_sim.item():.4f} - '
                f'Val MSE: {mse.item():.4f}'
            )
            wandb.log({'val/kl_div': kl_div.item(), 'val/cos_sim': cos_sim.item(), 'val/mse': mse.item()}, step=epoch)
            metric = kl_div.item() + 0.3 * mse.item() + 0.05 * cos_sim.item()
            if metric < best_metric:
                logger.info(f'Epoch {epoch + 1}/{config["epochs"]} - New best model!')
                best_metric = metric
                fabric.save(
                    f'{config["out_dir"]}/best_model.ckpt',
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                        "config": config,
                    },
                )


def test(fabric, model, test_loader, logger, config):
    checkpoint = fabric.load(f'{config["out_dir"]}/best_model.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    zs, preds, truths = [], [], []
    with torch.no_grad():
        test_tqdm = tqdm(
            enumerate(test_loader),
            total=len(test_loader),
            desc=f'Testing',
            disable=config['disable_tqdm']
        )
        for batch_idx, batch in test_tqdm:
            sns, lbs_phs, dur, _ = batch
            z, lbs_phs_hat = model(sns)
            zs.append(z)
            preds.append(lbs_phs_hat)
            truths.append(lbs_phs)

        zs = torch.cat(zs, dim=0)
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        log_preds = torch.nn.functional.log_softmax(preds, dim=1)
        preds = torch.nn.functional.softmax(log_preds, dim=1)
        kl_div = torch.nn.functional.kl_div(log_preds, truths, reduction='batchmean')
        cos_sim = torch.nn.functional.cosine_similarity(preds, truths, dim=1).mean()
        mse = torch.nn.functional.mse_loss(preds, truths)
        logger.info(
            f'Test KLDiv: {kl_div.item():.4f} - '
            f'Test CosSim: {cos_sim.item():.4f} - '
            f'Test MSE: {mse.item():.4f}'
        )
        np.save(f'{config["out_dir"]}/latent_encode.npy', zs.detach().cpu().numpy())
        df = pd.DataFrame({
            "Kullback-Leibler Divergence": [kl_div.item()],
            "Cosine Similarity": [cos_sim.item()],
            "Mean Squared Error": [mse.item()],
        })
        df.to_csv(f'{config["out_dir"]}/test_metric.csv', index=False, header=True)
        np.savetxt(f'{config["out_dir"]}/preds_soft_lbs.txt', preds.detach().cpu().numpy(), fmt='%.4f')
        np.savetxt(f'{config["out_dir"]}/truths_soft_lbs.txt', truths.detach().cpu().numpy(), fmt='%.4f')


def train_hypnos(fabric, model, dataset, logger, config):
    train_config = {**config['train'], "out_dir": config['out_dir']}
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        mode='online',
        project="HypnosNet",
        dir=Path(config['logs']['log_dir']).parent.absolute(),
        config=config['train'],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])
    train_loader, val_loader, test_loader = get_loader(dataset, train_config)

    # setup to fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    fit(fabric, model, train_loader, val_loader, optimizer, logger, train_config)
    test(fabric, model, test_loader, logger, train_config)

    wandb.finish()
