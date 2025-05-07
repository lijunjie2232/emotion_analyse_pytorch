import torch
import torch.nn as nn
from layers import Conv, C3k2, A2C2f, Classify


# ## build model
class EmotionNet(nn.Module):
    def __init__(self, c1: int = 3, nc: int = 7):
        """
        Initialize classification model

        Args:
            c1 (int): Input channel size
            nc (int): Number of output classes
        """
        super().__init__()

        # Calculate scaling parameters from config

        # Build backbone
        self.backbone = nn.ModuleList(
            [
                # 0-P1/2
                Conv(c1=3, c2=16, k=3, s=2),
                # 1-P2/4
                Conv(c1=16, c2=32, k=3, s=2),
                # 2-C3k2 block
                C3k2(c1=32, c2=64, n=1, e=0.25),
                # 3-P3/8
                Conv(c1=64, c2=128, k=3, s=2),
                # 4-C3k2 block
                C3k2(c1=128, c2=128, n=1, e=0.25),
                # 5-P4/16
                Conv(c1=128, c2=128, k=3, s=2),
                # 6-A2C2f block
                A2C2f(c1=128, c2=128, n=2, a2=True, area=4, e=0.5),
                # 7-P5/32
                Conv(c1=128, c2=256, k=3, s=2),
                # 8-A2C2f block
                A2C2f(c1=256, c2=256, n=2, a2=True, area=1, e=0.5),
            ]
        )

        # Build classification head
        self.classify = Classify(256, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Pass through backbone
        for layer in self.backbone:
            x = layer(x)

        # Pass through classification head
        x = self.classify(x)

        return x
