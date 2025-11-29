import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # --- CONVOLUTIONAL BLOCKS ---
        
        # Block 1 (Input 3 -> Output 32)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1) # Added padding=1 to preserve dimensions
        self.gn1 = nn.GroupNorm(1,32)

        # Block 2 (Input 32 -> Output 64)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.gn2 = nn.GroupNorm(1,64)

        # Block 3 (Input 64 -> Output 128)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.gn3 = nn.GroupNorm(1,128)

        # --- OTHER LAYERS ---
        self.pool = nn.MaxPool2d(2, 2)
        self.adapt = nn.AdaptiveAvgPool2d((4,4))

        # Defined Dropout layer for the final classification step
        self.dropout = nn.Dropout(p=0.5) 

        # Final classifier (Input: 128 * 4*4 = 2048 features)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        # Conv1 -> BN -> Activation (GELU) -> Pool
        x = self.pool(F.gelu(self.gn1(self.conv1(x))))
        
        # Conv2 -> BN -> Activation (GELU) -> Pool
        x = self.pool(F.gelu(self.gn2(self.conv2(x))))
        
        # Conv3 -> BN -> Activation (GELU) -> Pool
        x = self.pool(F.gelu(self.gn3(self.conv3(x))))

        # Shrink to fixed 4×4 regardless of input size
        x = self.adapt(x)

        # Flatten features
        x = torch.flatten(x, 1)

        # Apply Dropout before the final Linear classifier
        # x = self.dropout(x) 

        x = self.fc(x)

        return x
