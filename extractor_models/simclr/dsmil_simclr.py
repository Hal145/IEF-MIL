import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img



def dsmil_simclr():

    pretrain = False
    norm_layer = 'instance'
    resnet = models.resnet18(pretrained=pretrain, norm_layer=norm_layer)
    num_feats = 512
    resnet.fc = nn.Identity()

    i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=4)
    weight_path = os.path.join('saved_models', 'milc16', '20x', 'model-v0.pth')

    state_dict_weights = torch.load(weight_path)
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier_h.state_dict()
    new_state_dict = OrderedDict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
    new_state_dict[name] = v
    i_classifier_h.load_state_dict(new_state_dict, strict=False)
    print('Using pretrained weights of DSMIL 20x SimCLR')

    return i_classifier_h

