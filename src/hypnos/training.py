import torch
from tqdm import tqdm

from src.hypnos.train_utils import cal_kl_mse_cos_entropy_loss


def fit(fabric, model, train_loader, val_loader, optimizer, logger, config):
    model.train()
    for epoch in range(config['train']['epochs']):
        logger.info(f'Epoch {epoch + 1}/{config["train"]["epochs"]}')
        total_loss = total_samples = 0
        train_tqdm = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Training',
            disable=config['train']['disable_tqdm']
        )
        for batch_idx, batch in train_tqdm:
            optimizer.zero_grad()
            sns, lbs_phs, dur, _ = batch
            z, lbs_phs_hat = model(sns)
            loss = cal_kl_mse_cos_entropy_loss(lbs_phs_hat, lbs_phs, config['train']['kl_t'])
            fabric.backward(loss)

            # Stuck-Survival Training (Meta AI 2022)
            for n, p in model.named_parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * config['train'].get('sst_noise', 1e-4)
                    # logger.debug(f'Gradient norm after SST of {n}: {p.grad.norm().item():.4f}')

            optimizer.step()
            total_loss += loss.item()
            total_samples += lbs_phs.shape[0]
            train_tqdm.set_postfix(loss=f'{total_loss / total_samples:.4f}')
        logger.info(f'Epoch {epoch + 1}/{config["train"]["epochs"]} - Train Loss: {total_loss / total_samples:.4f}')

        model.eval()
        val_loss = val_samples = 0
        with torch.no_grad():
            val_tqdm = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc='Validating',
                disable=config['train']['disable_tqdm']
            )
            for batch_idx, batch in val_tqdm:
                sns, lbs_phs, dur, _ = batch
                z, lbs_phs_hat = model(sns)
                loss = cal_kl_mse_cos_entropy_loss(lbs_phs_hat, lbs_phs, config['train']['kl_t'])

                if batch_idx == 0:
                    logger.debug(f'lbs_phs_hat: {torch.softmax(lbs_phs_hat, dim=1)}')

                val_loss += loss.item()
                val_samples += lbs_phs.shape[0]
                val_tqdm.set_postfix(val_loss=f'{val_loss / val_samples:.4f}')
        logger.info(f'Epoch {epoch + 1}/{config["train"]["epochs"]} - Val Loss: {val_loss / val_samples:.4f}')


def test():
    pass
