import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm

from src.hypnos.data_loader import get_loader
from src.hypnos.train_utils import cal_kl_mse_cos_entropy_alignment_loss, cal_eval_metrics


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
            sns, lbs, lbs_vec = batch
            logits, lbs_align = model(sns, lbs_vec)
            kl_t = 1 + config['kl_e'] / (epoch + config['kl_e'])
            loss = cal_kl_mse_cos_entropy_alignment_loss(logits, lbs_align, lbs, kl_t, align_l=0.3)
            fabric.backward(loss)

            # Stuck-Survival Training (Meta AI 2022)
            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * config.get('sst_noise', 1e-4)
                    # logger.debug(f'Gradient norm after SST of {n}: {p.grad.norm().item():.4f}')

            optimizer.step()
            total_loss += loss.item()
            total_samples += lbs.shape[0]
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
                sns, lbs, _ = batch
                _, logits = model.predict_raw(sns)
                preds.append(logits)
                truths.append(lbs)

            preds = torch.cat(preds, dim=0)
            truths = torch.cat(truths, dim=0)
            val_auroc, val_ap, val_f1x = cal_eval_metrics(preds, truths)
            logger.info(f"Val AUROC: {val_auroc:.4f} - Val AP: {val_ap:.4f} - Val F1X: {val_f1x:.4f}")
            wandb.log({'val/auroc': val_auroc, 'val/ap': val_ap, 'val/f1x': val_f1x}, step=epoch)

            if val_f1x < best_metric:
                logger.info(f'Epoch {epoch + 1}/{config["epochs"]} - New best model!')
                best_metric = val_f1x
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
            sns, lbs, _ = batch
            z, logits = model.predict_raw(sns)
            zs.append(z)
            preds.append(logits)
            truths.append(lbs)

        zs = torch.cat(zs, dim=0)
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)
        test_auroc, test_ap, test_f1x = cal_eval_metrics(preds, truths)
        logger.info(f"Test AUROC: {test_auroc:.4f} - Test AP: {test_ap:.4f} - Test F1X: {test_f1x:.4f}")
        np.save(f'{config["out_dir"]}/latent_encode.npy', zs.detach().cpu().numpy())
        df = pd.DataFrame({
            "AUROC": [test_auroc],
            "AP": [test_ap],
            "F1X": [test_f1x],
        })
        df.to_csv(f'{config["out_dir"]}/test_metric.csv', index=False, header=True)
        np.savetxt(f'{config["out_dir"]}/preds_soft_lbs.txt', preds.detach().cpu().numpy(), fmt='%.4f')
        np.savetxt(f'{config["out_dir"]}/truths_soft_lbs.txt', truths.detach().cpu().numpy(), fmt='%.4f')


def train_hypnos(fabric, model, dataset, logger, config):
    train_config = {
        **config['train'],
        "out_dir": config['out_dir'],
        "processed_data_dir": config['data']['processed_data_dir']
    }
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
