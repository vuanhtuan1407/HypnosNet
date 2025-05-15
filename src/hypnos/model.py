import numpy as np
import torch
from torch import nn

from src.hypnos.params import LB_MAT


class HypnosNet(nn.Module):
    def __init__(self, win_len=256, hop_len=64, dropout=0.1, cnn_outdim=8, emb_dim=128):
        super().__init__()
        self.encoder = Encoder(win_len, hop_len, dropout, emb_dim, cnn_outdim)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(emb_dim * 4, 3),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 7),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = x[:, :, 0]  # only use the eeg value
        x = self.encoder(x)
        soft_cls = self.classifier(x)
        factor = torch.tensor(np.array(LB_MAT), dtype=torch.float32, device=x.device)
        hard_cls = torch.matmul(soft_cls, factor.transpose(0, 1))
        # hard_cls = self.decoder(soft_cls)
        return soft_cls, hard_cls

    def predict_soft(self, x):
        x = x[:, :, 0]
        x = self.encoder(x)
        cls_logits = self.classifier(x)
        return x, cls_logits

    def predict_hard(self, x):
        x = x[:, :, 0]
        x = self.encoder(x)
        soft_cls = self.classifier(x)
        # hard_cls = self.decoder(cls_logits)
        factor = torch.tensor(np.array(LB_MAT), dtype=torch.float32, device=x.device)
        hard_cls = torch.matmul(soft_cls, factor.transpose(0, 1))
        return x, hard_cls


class Encoder(nn.Module):
    def __init__(self, win_len=256, hop_len=64, dropout=0.1, emb_dim=128, cnn_outdim=1):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.emb_dim = emb_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.p_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.p_conv2 = nn.Sequential(
            nn.Conv2d(64, cnn_outdim, kernel_size=1),
            nn.BatchNorm2d(cnn_outdim)
        )

        # Fully connected layers with ReLU
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * cnn_outdim, 32 * 4 * cnn_outdim * 4),
            nn.ReLU(),
            nn.Linear(32 * 4 * cnn_outdim * 4, emb_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        z = torch.stft(
            x,
            n_fft=self.win_len,
            hop_length=self.hop_len,
            window=torch.hamming_window(self.win_len, device=x.device),
            return_complex=True,
            # normalized=True,
            onesided=True
        )
        # z = torch.log1p(torch.abs(z) + 1e-9)
        z = 20 * torch.log10(torch.clamp(torch.abs(z), min=1e-6))  # magnitude -> amplitude (dB)
        z = z.unsqueeze(1)  # shape: [B, 1, F, T]
        z = torch.nn.functional.relu(self.conv1(z) + z)
        z = self.maxpool1(z)
        z = self.p_conv1(z)
        z = torch.nn.functional.relu(self.conv2(z) + z)
        z = self.maxpool2(z)
        z = self.p_conv2(z)
        z = z.reshape(z.shape[0], -1)
        z = self.fc(z)
        return z
