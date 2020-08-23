from __future__ import print_function
import onnxruntime
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
# from main import SRModel
import numpy as np
import matplotlib.pyplot as plt


# Training settings
# parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# parser.add_argument('--input_image', type=str,
#                     required=True, help='input image to use')
# parser.add_argument('--model', type=str, required=True,
#                     help='model file to use')
# parser.add_argument('--output_filename', type=str,
#                     help='where to save the output image')
# parser.add_argument('--cuda', action='store_true', help='use cuda')
# opt = parser.parse_args()

# print(opt)
input_image = "/Users/ajaichemmanam/Documents/Projects/PytorchLightning/SR/dataset/BSDS300/images/test/3096.jpg"
output_file = "/Users/ajaichemmanam/Documents/Projects/PytorchLightning/SR/output.jpg"
# modelPath = "/Users/ajaichemmanam/Documents/Projects/PytorchLightning/SR/lightning_logs/version_0/checkpoints/epoch=1.ckpt"

img = Image.open(input_image)
img = img.resize((224, 224))
img_ycbcr = img.convert('YCbCr')
y, cb, cr = img_ycbcr.split()
img_ndarray = np.asarray(y)
img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
imginput = img_4.astype(np.float32) / 255.0
print(imginput.shape)

ort_session = onnxruntime.InferenceSession('model.onnx')
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: imginput}
ort_outs = ort_session.run(None, ort_inputs)
ort_outs = np.array(ort_outs)
ort_outs = ort_outs.squeeze(0)
ort_outs = ort_outs.squeeze(0)
print(ort_outs.shape)
ort_outs *= 255.0
ort_outs = ort_outs.clip(0, 255)
ort_outs = Image.fromarray(np.uint8(ort_outs[0]), mode='L')

out_img = Image.merge(
    "YCbCr", [
        ort_outs,
        cb.resize(ort_outs.size, Image.BICUBIC),
        cr.resize(ort_outs.size, Image.BICUBIC),
    ]).convert("RGB")
plt.imshow(out_img)
out_img.save(output_file)
print('output image saved to ', output_file)
