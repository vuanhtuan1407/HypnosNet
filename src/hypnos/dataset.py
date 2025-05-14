import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset


# class EEGDataset(Dataset):
#     def __init__(self, file):
#         self.data = joblib.load(file)
#         self.eeg = torch.tensor(self.data[0], dtype=torch.float32)
#         self.lbs = self.data[1]
#         lbs_phs, dur, trans_len = [], [], []
#         for lb in self.lbs:
#             lbs_phs.append(lb[0])
#             dur.append(lb[1])
#             trans_len.append(lb[2])
#
#         self.lbs_phs = torch.tensor(np.array(lbs_phs), dtype=torch.float32)
#         self.dur = torch.tensor(np.array(dur), dtype=torch.int32)
#         self.trans_len = torch.tensor(np.array(trans_len), dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.eeg)
#
#     def __getitem__(self, idx):
#         return self.eeg[idx], self.lbs_phs[idx], self.dur[idx], self.trans_len[idx]
#
#     def get_data(self):
#         return self.data, self.lbs

class EEGDataset(Dataset):
    def __init__(self, file):
        self.data = joblib.load(file)
        sns, lbs, lbs_onehot, lbs_vec = [], [], [], []
        for (sn, lb, lb_vec) in self.data:
            sns.append(sn)
            lb_onehot = torch.zeros(7, dtype=torch.float32)
            lb_idx = lb % 6 if lb != -1 else 6
            lb_onehot[lb_idx] = 1.0
            lbs_onehot.append(lb_onehot)
            lbs.append(lb_idx)
            lbs_vec.append(lb_vec)

        self.sns = torch.tensor(np.array(sns), dtype=torch.float32)
        self.lbs = torch.tensor(np.array(lbs), dtype=torch.int32)
        self.lbs_onehot = torch.tensor(np.array(lbs_onehot), dtype=torch.float32)
        self.lbs_vec = torch.tensor(np.array(lbs_vec), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.sns[idx], self.lbs[idx], self.lbs_onehot[idx], self.lbs_vec[idx]

    def get_data(self):
        return self.data


def get_dataset(files):
    dts = []
    for file in files:
        dt = EEGDataset(file)
        dts.append(dt)
    return ConcatDataset(dts)


if __name__ == "__main__":
    dataset = EEGDataset('../../data/processed/data_0.pkl')
    print(dataset.__getitem__(1))
