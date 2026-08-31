import torch
from torch import nn


class CNNTemporal(nn.Module):

    def __init__(
        self,
        n_features=13,
        n_classes=4,
        dropout=0.25
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                n_features,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),


            nn.Conv1d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),


            nn.Conv1d(
                64,
                96,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(96),
            nn.ReLU(),


            nn.Conv1d(
                96,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )


        self.pool = nn.AdaptiveAvgPool1d(1)


        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,n_classes)
        )


    def forward(self,x):

        # x:
        # batch,60,13

        x=x.transpose(1,2)

        x=self.features(x)

        x=self.pool(x)

        x=x.squeeze(-1)

        x=self.classifier(x)

        return x