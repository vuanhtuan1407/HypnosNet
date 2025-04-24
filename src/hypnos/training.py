import torch
from tqdm import tqdm


def fit(fabric, model, train_loader, val_loader, optimizer, config):
    model.train()
    for epoch in range(config['epochs']):
        print(f'Epoch {epoch}')
        total_loss = 0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training'):
            optimizer.zero_grad()
            sns, lbs_phs, dur, _ = batch
            z, lbs_phs_hat = model(sns)
            loss = torch.nn.functional.cross_entropy(lbs_phs_hat, lbs_phs)
            fabric.backward(loss)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch} loss: {total_loss / len(train_loader)}')


def test():
    pass
