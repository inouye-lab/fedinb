import torch.nn as nn

class AE(nn.Module):
    def __init__(self, dataset):
        super(AE, self).__init__()
        if dataset == 'rmnist':
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

            self.decoder = nn.Sequential(
                nn.Conv2d(8, 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid()
            )
        elif dataset == 'rfmnist':
            # https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 8, 3, stride=2, padding=1),
                nn.ReLU(True),
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Conv2d(16, 32, 3, stride=2, padding=0),
                nn.ReLU(True)
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3,
                                   stride=2, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.Sigmoid()
            )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x