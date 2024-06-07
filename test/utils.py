import torch
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join


def load_images(data_dir, dims):
    filenames = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    img_list = []
    for filename in filenames:
        img_list.append(img2tensor(join(data_dir, filename), dims, expand=True))
    return torch.cat(img_list, dim=0)


def img2tensor(data_path, desired_size=None, expand=False, view=False):
    img = Image.open(data_path)
    img = img.convert('RGB')
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = torch.from_numpy(np.asarray(img, dtype='float32'))
    if expand:
        x = x.unsqueeze(0)  # add batch dim
    x /= 255.0
    return x


def tensor2img(x):
    x = x.detach().numpy()

    # ensure the tensor is in the range [0, 1]
    x = x + max(-x.min(), 0)
    x_max = x.max()
    if x_max != 0:
        x /= x_max
    
    # scale the tensor to the range [0, 255]
    x *= 255

    return Image.fromarray(x.astype('uint8'), 'RGB')


def deg2rad(x):
    return (x * np.pi) / 180
