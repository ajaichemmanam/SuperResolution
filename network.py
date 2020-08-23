
import torch
import torch.nn as nn
import torch.nn.init as init

import torch.optim as optim
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
from data import get_3Dtraining_set, get_3Dtest_set

import pytorch_lightning as pl

from math import log10


class SRModel(pl.LightningModule):

    def __init__(self, upscale_factor):
        super(SRModel, self).__init__()
        self.upscaleFactor = upscale_factor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# Done
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        train_set = get_training_set(self.upscaleFactor)
        training_data_loader = DataLoader(
            dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
        return training_data_loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        result = pl.TrainResult(loss)
        return result

    # def val_dataloader(self):
    #     mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
    #     return mnist_val

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     result = pl.EvalResult(checkpoint_on=loss)
    #     result.log('val_loss', loss)
    #     return result

    def test_dataloader(self):
        test_set = get_test_set(self.upscaleFactor)

        testing_data_loader = DataLoader(
            dataset=test_set, num_workers=4, batch_size=10, shuffle=False)

        return testing_data_loader

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.MSELoss()
        loss = criterion(y_hat, y)
        psnr = 10 * log10(1 / loss)
        result = pl.EvalResult()
        result.log('test_loss', loss)
        result.log('test_psnr', psnr)
        return result

