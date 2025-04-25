import torch


def cal_masked_ce_loss():
    pass


def cal_sup_con_loss():
    pass


def cal_kl_mse_cos_entropy_loss(lbs_phs_hat, lbs_phs, kl_t):
    log_probs = torch.nn.functional.log_softmax(lbs_phs_hat, dim=1)
    probs = torch.nn.functional.softmax(lbs_phs_hat / kl_t, dim=1)
    kl = torch.nn.functional.kl_div(log_probs, lbs_phs, reduction='batchmean') * (kl_t ** 2)
    mse = torch.nn.functional.mse_loss(probs, lbs_phs)
    cos = 1 - torch.nn.functional.cosine_similarity(probs, lbs_phs, dim=1).mean()
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1).mean()
    return kl + 0.3 * mse + 0.05 * cos + 0.01 * entropy
