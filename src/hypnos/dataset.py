import joblib
import numpy as np
from torch.utils.data import Dataset
import torch


class EEGDataset(Dataset):
    def __init__(self, file):
        self.data = joblib.load(file)
        self.eeg = torch.tensor(self.data[0], dtype=torch.float32)
        self.lbs = self.data[1]
        lbs_phs, dur, trans_len = [], [], []
        for lb in self.lbs:
            lbs_phs.append(lb[0])
            dur.append(lb[1])
            trans_len.append(lb[2])

        self.lbs_phs = torch.tensor(np.array(lbs_phs), dtype=torch.float32)
        self.dur = torch.tensor(np.array(dur), dtype=torch.int32)
        self.trans_len = torch.tensor(np.array(trans_len), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.eeg[idx], self.lbs_phs[idx], self.dur[idx], self.trans_len[idx]

    def get_data(self):
        return self.data, self.lbs


if __name__ == "__main__":
    dataset = EEGDataset('../../data/processed/K3_EEG3_11h.pkl')
    print(dataset.__getitem__(1))
