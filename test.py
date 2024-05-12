import torch
import torchvision

print(torch.__version__)
# '1.13.0+cu117'

print(torch.cuda.is_available())
# True

print(torchvision.__version__)
# '0.14.0+cu117'