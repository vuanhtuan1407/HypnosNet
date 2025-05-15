import numpy as np
import torch
from sklearn.metrics import average_precision_score as auprc
from sklearn.metrics import roc_auc_score as auroc
from torch.utils.data import DataLoader

from src.hypnos.model import HypnosNet
from src.hypnos.params import LB_VEC, LB_MAT
from src.hypnos.utils import log


def cal_ce_entropy_loss(hard_lbs_hat, hard_lbs):
    assert hard_lbs_hat.shape == hard_lbs.shape
    assert hard_lbs_hat.shape[-1] == 7
    hard_lbs_hat = torch.nn.functional.softmax(hard_lbs_hat, dim=-1)
    hard_lbs = torch.argmax(hard_lbs, dim=-1)
    ce = torch.nn.functional.cross_entropy(hard_lbs_hat, hard_lbs, ignore_index=6)
    entropy = -(hard_lbs_hat * hard_lbs_hat.clamp(min=1e-8).log()).sum(dim=1).mean()
    return ce + 0.01 * entropy


def cal_kl_mse_cos_entropy_loss(soft_lbs_hat, soft_lbs, kl_t):
    log_probs = torch.nn.functional.log_softmax(soft_lbs_hat / kl_t, dim=1)
    probs = torch.nn.functional.softmax(soft_lbs_hat, dim=1)
    kl = torch.nn.functional.kl_div(log_probs, soft_lbs, reduction='batchmean') * (kl_t ** 2)
    mse = torch.nn.functional.mse_loss(probs, soft_lbs)
    cos = 1 - torch.nn.functional.cosine_similarity(probs, soft_lbs, dim=1).mean()
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
    return kl + 0.3 * mse + 0.05 * cos + 0.01 * entropy


def cal_fit_hypnos_loss(hard_lbs_hat, soft_lbs_hat, hard_lbs, sorf_lbs, kl_t, lambda_ce=1, lambda_kt=0.5):
    return (lambda_ce * cal_ce_entropy_loss(hard_lbs_hat, hard_lbs)
            + lambda_kt * cal_kl_mse_cos_entropy_loss(soft_lbs_hat, sorf_lbs, kl_t))


def cal_eval_metrics(hard_lbs_hat, hard_lbs):
    hard_lbs_hat = torch.nn.functional.softmax(hard_lbs_hat, dim=-1).detach().cpu().numpy()
    hard_lbs = torch.argmax(hard_lbs, dim=-1).detach().cpu().numpy()
    auroc_score = auroc(hard_lbs, hard_lbs_hat, multi_class='ovr')
    ap_score = auprc(hard_lbs, hard_lbs_hat)
    f1x = 2 * auroc_score * ap_score / (auroc_score + ap_score + 1e-9)
    return auroc_score, ap_score, f1x


def get_model(model_name, logger=None):
    if model_name == 'hypnos':
        return HypnosNet()
    else:
        log("Model not supported! Use default model.", logger, 'error')
        return HypnosNet()


def get_loader(train_set, val_set, test_set, config):
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config['batch_size'],
        shuffle=True,
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
