import numpy as np
import torch
from sklearn.metrics import roc_auc_score as auroc
from sklearn.metrics import average_precision_score as auprc

from src.hypnos.params import LB_VEC, LB_MAT


def cal_masked_ce_loss():
    pass


def cal_sup_con_loss():
    pass


def cal_kl_mse_cos_entropy_alignment_loss(logits, lbs_align, lbs, kl_t, align_l=0.1):
    log_probs = torch.nn.functional.log_softmax(logits / kl_t, dim=1)
    probs = torch.nn.functional.softmax(logits, dim=1)
    kl = torch.nn.functional.kl_div(log_probs, lbs_align, reduction='batchmean') * (kl_t ** 2)
    mse = torch.nn.functional.mse_loss(probs, lbs_align)
    cos = 1 - torch.nn.functional.cosine_similarity(probs, lbs_align, dim=1).mean()
    entropy = -(probs.clamp(min=1e-8) * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
    init_lbs_vec = __get_init_lbs_vec(lbs)
    alignment = torch.nn.functional.mse_loss(lbs_align, init_lbs_vec)
    return kl + 0.3 * mse + 0.05 * cos + 0.01 * entropy + align_l * alignment


def __get_init_lbs_vec(lbs):
    init_lbs_vec = []
    for lb in lbs:
        init_lbs_vec.append(LB_VEC[lb.item()])
    return torch.tensor(init_lbs_vec, dtype=torch.float32, device=lbs.device)


def cal_eval_metrics(logits, lbs):
    # ce = torch.nn.functional.cross_entropy(logits, lbs)
    lbs_mat = torch.tensor(np.array(LB_MAT), dtype=torch.float32, device=logits.device)
    logits = torch.matmul(logits, lbs_mat.transpose(0, 1))
    lbs_hat = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
    lbs = lbs.detach().cpu().numpy()
    auroc_score = auroc(lbs, lbs_hat, multi_class='ovr')
    ap_score = auprc(lbs, lbs_hat)
    f1x = 2 * auroc_score * ap_score / (auroc_score + ap_score + 1e-9)
    return auroc_score, ap_score, f1x
