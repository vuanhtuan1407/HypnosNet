import torch
from torch import nn


class HypnosNet(nn.Module):
    def __init__(self, win_len=256, hop_len=64, dropout=0.2, cnn_outdim=1, emb_dim=128):
        super().__init__()
        self.encoder = Encoder(win_len, hop_len, dropout, emb_dim, cnn_outdim)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(emb_dim * 2, 3),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.encoder(x)
        cls_logits = self.classifier(x)
        return x, cls_logits


class Encoder(nn.Module):
    def __init__(self, win_len=256, hop_len=64, dropout=0.2, emb_dim=128, cnn_outdim=1):
        super().__init__()
        self.win_len = win_len
        self.hop_len = hop_len
        self.emb_dim = emb_dim

        # CNN layers (fixed)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.p_conv1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(1024, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.p_conv2 = nn.Sequential(
            nn.Conv2d(256, cnn_outdim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * cnn_outdim, 32 * 4 * cnn_outdim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32 * 4 * cnn_outdim * 2, 32 * 4 * cnn_outdim * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32 * 4 * cnn_outdim * 4, emb_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        z = torch.stft(
            x,
            n_fft=self.win_len,
            hop_length=self.hop_len,
            window=torch.hamming_window(self.win_len, device=x.device),
            return_complex=True,
            normalized=False,
            onesided=True
        )
        z = torch.abs(z)
        z = z.unsqueeze(1)
        z = self.conv1(z) + z
        z = self.maxpool1(z)
        z = self.p_conv1(z)
        z = self.conv2(z) + z
        z = self.maxpool2(z)
        z = self.p_conv2(z)
        z = z.reshape(z.shape[0], -1)
        z = self.fc(z)
        return z
