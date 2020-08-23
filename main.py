from __future__ import print_function
import argparse
from math import log10

import torch
from network import SRModel

import pytorch_lightning as pl

torch.manual_seed(123)
# init model
model = SRModel(upscale_factor=2)
model.example_input_array = torch.randn((1, 1, 224, 224))
# Train
trainer = pl.Trainer(max_epochs=500, fast_dev_run=False)
trainer.fit(model)

filepath = 'model.onnx'
# input_sample = torch.randn((1, 1, 224, 224))
# model.to_onnx(filepath, input_sample, export_params=True)
model.to_onnx(filepath, export_params=True)
print("Exported to onnx")
