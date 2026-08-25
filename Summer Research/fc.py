import torch.nn as nn


class FCBlock(nn.Module):
    def __init__(self, input_dim, dropout, num_outputs, scale=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim*scale),
            nn.BatchNorm1d(input_dim*scale),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(input_dim*scale, input_dim*scale//2),
            nn.BatchNorm1d(input_dim*scale//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(input_dim*scale//2, input_dim*scale//4),
            nn.BatchNorm1d(input_dim*scale//4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),


            nn.Linear(input_dim*scale//4, num_outputs))

    def forward(self, x):
        return self.block(x)


class GalaxyClassifier(nn.Module):
    "Full Galaxy Classifier network."
    def __init__(
        self,
        num_outputs: int = 1,
        input_dim: int = 8,
        scale: int = 4,
        dropout: float = 0.4,
        **kwargs):

        super().__init__()
        self.fc = FCBlock(
            input_dim=input_dim, 
            dropout=dropout, 
            num_outputs=num_outputs, 
            scale=scale)

    def forward(self, x):
        x = self.fc(x)

        return x