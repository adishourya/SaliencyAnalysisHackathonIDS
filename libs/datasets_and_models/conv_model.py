import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # simple 3-layer conv stack
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # final adaptive pool → 4x4
        self.adapt = nn.AdaptiveAvgPool2d((2, 2))

        # final classifier (only 64*2*2 = 256 features)
        self.fc = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 224→112
        x = self.pool(F.relu(self.conv2(x)))   # 112→56
        x = self.pool(F.relu(self.conv3(x)))   # 56→28

        # shrink to fixed 4×4 regardless of input size
        x = self.adapt(x)

        x = torch.flatten(x, 1)                # 64*2*2*37
        x = self.fc(x)

        return x

