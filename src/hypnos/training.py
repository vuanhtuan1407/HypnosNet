import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.hypnos.train_utils import cal_eval_metrics, get_loader, get_model, cal_fit_hypnos_loss
from src.hypnos.utils import log


def fit_hypnos(fabric, model, train_loader, val_loader, optimizer, config, wandb, logger=None):
    best_metric = 1e-6
    for epoch in range(config['epochs']):
        model.train()
        total_loss = total_samples = 0
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training Epoch {epoch + 1}/{config["epochs"]}',
            disable=config['disable_tqdm']
        )
        for batch_idx, batch in train_tqdm:
            optimizer.zero_grad()
            sns, lbs, lbs_onehot, lbs_vec = batch
            soft_lbs_hat, hard_lbs_hat = model(sns)
            kl_t = 1 + config['kl_e'] / (epoch + config['kl_e'])
            loss = cal_fit_hypnos_loss(hard_lbs_hat, soft_lbs_hat, lbs_onehot, lbs_vec, kl_t, 0.1, 1.0)
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
        log(f'Epoch {epoch + 1}/{config["epochs"]} - Train Loss: {total_loss / total_samples:.4f}', logger)
        wandb.log({'train/loss': total_loss / total_samples}, step=epoch)

        model.eval()
        # val_loss = val_samples = 0
        preds, truths = [], []
        with torch.no_grad():
            val_tqdm = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f'Validating Epoch {epoch + 1}/{config["epochs"]}',
                disable=config['disable_tqdm']
            )
            for batch_idx, batch in val_tqdm:
                sns, _, lbs_onehot, _ = batch
                _, logits = model.predict_hard(sns)
                preds.append(logits)
                truths.append(lbs_onehot)

            preds = torch.cat(preds, dim=0)
            truths = torch.cat(truths, dim=0)

            # valid_mask = truths[:, :-1].sum(dim=-1) > 0
            # preds = preds[:, :-1][valid_mask]
            # truths = truths[:, :-1][valid_mask]
            val_auroc, val_ap, val_f1x = cal_eval_metrics(preds, truths)
            log(f"Val AUROC: {val_auroc:.4f} - Val AP: {val_ap:.4f} - Val F1X: {val_f1x:.4f}", logger)
            wandb.log({'val/auroc': val_auroc, 'val/ap': val_ap, 'val/f1x': val_f1x}, step=epoch)

            if val_f1x > best_metric:
                log(f'Epoch {epoch + 1}/{config["epochs"]} - New best model!', logger)
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


def test(fabric, model, test_loader, config, logger=None):
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
            sns, _, lbs_onehot, _ = batch
            z, logits = model.predict_hard(sns)
            zs.append(z)
            preds.append(logits)
            truths.append(lbs_onehot)

        zs = torch.cat(zs, dim=0)
        preds = torch.cat(preds, dim=0)
        truths = torch.cat(truths, dim=0)

        # valid_mask = truths[:, :-1].sum(dim=-1) > 0
        # preds = preds[:, :-1][valid_mask]
        # truths = truths[:, :-1][valid_mask]
        test_auroc, test_ap, test_f1x = cal_eval_metrics(preds, truths)
        log(f"Test AUROC: {test_auroc:.4f} - Test AP: {test_ap:.4f} - Test F1X: {test_f1x:.4f}", logger)
        np.save(f'{config["out_dir"]}/latent_encode.npy', zs.detach().cpu().numpy())
        df = pd.DataFrame({
            "AUROC": [test_auroc],
            "AP": [test_ap],
            "F1X": [test_f1x],
        })
        df.to_csv(f'{config["out_dir"]}/test_metric.csv', index=False, header=True)
        np.savetxt(f'{config["out_dir"]}/preds_soft_lbs.txt', preds.detach().cpu().numpy(), fmt='%.4f')
        np.savetxt(f'{config["out_dir"]}/truths_soft_lbs.txt', truths.detach().cpu().numpy(), fmt='%.4f')


def train_model(fabric, model_name, train_dataset, val_dataset, test_dataset, config, wandb, logger=None):
    train_config = config['train']
    model = get_model(model_name, logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"])
    train_loader, val_loader, test_loader = get_loader(train_dataset, val_dataset, test_dataset, train_config)

    # setup to fabric
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader, test_loader = fabric.setup_dataloaders(train_loader, val_loader, test_loader)

    # training & testing
    if train_config['model_name'] == 'hypnos':
        fit_hypnos(fabric, model, train_loader, val_loader, optimizer, train_config, wandb, logger)
    test(fabric, model, test_loader, train_config, logger)
