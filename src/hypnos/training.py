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
            optimizer.step()
            total_loss += loss.item()
            total_samples += lbs_phs.shape[0]
            train_tqdm.set_postfix(loss=f'{total_loss / total_samples:.4f}')
        logger.info(f'Epoch {epoch + 1}/{config["train"]["epochs"]} - Train Loss: {total_loss / total_samples:.4f}')


def test():
    pass
