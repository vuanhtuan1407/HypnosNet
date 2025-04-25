import torch


def cal_masked_ce_loss():
    pass


def cal_sup_con_loss():
    pass


def cal_kl_mse_loss(lbs_phs_hat, lbs_phs, kl_t):
    lbs_phs_hat = torch.nn.functional.softmax(lbs_phs_hat / kl_t, dim=1)
    kl = torch.nn.functional.kl_div(torch.log(lbs_phs_hat), lbs_phs, reduction='batchmean') * (kl_t ** 2)
    mse = torch.nn.functional.mse_loss(lbs_phs_hat, lbs_phs)
    loss = kl + 0.5 * mse
    return loss
