import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os
from tqdm import tqdm
import math
from rich import traceback, pretty, inspect, print
from shape_classifier.pl.lightning_module import LitModel
from torchvision import transforms
import random
# from visdom import Visdom
# port = 8097
# server = "127.0.0.1"
# base_url = '/'
# username = ''
# password = ''
# vis = Visdom(port=port, server=server, base_url=base_url)
# assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'

data_path = "/scratch/alotaima/datasets/v6-shape"
output_path ="/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/vis/aug"
set_names = ['img_id', 'vis', 'class_num']
old_classes = [
    'cube',
    'cone',
    'circle frustum',
    'cylinder',
    'pyramid',
    'square frustum',
    'letter l',
    'triangular prism',
    'car',
    'duck',
    'sphere',
    'train',
    'trolley',
    'tube narrow',
    'tube wide',
    'turtle',
    'pole',
]
classes = [
    "cube",
    "pole",
    "else"
]
df = pd.read_csv(f"{data_path}/test/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
set_ = (
    df
    [df['vis'] >= 0.25] # filter
    .reset_index()
)
print(len(set_['class_num']))
print(set_.count())
groups = (
    set_
    .groupby(['class_num']) # grouping
    # ['class_num']
    # .apply(list)
    # .to_dict()
    .groups
)
print(groups.keys())
print([len(group) for _, group in groups.items()])
new_groups = {
    0: [],
    1: [],
    2: []
}
for i, group in groups.items():
    if i == 0:
        new_groups[0].extend(group)
    elif i > 0 and i < 16:
        new_groups[2].extend(group)
    elif i == 16:
        new_groups[1].extend(group)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transform(train: bool):
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.Resize((224,224)),             
            transforms.ToTensor(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.5),
            transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.25),  
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([      
        transforms.ToTensor(),
        transforms.Resize((224,224)),     
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
trans_train = get_transform(True)
trans_val = get_transform(False)
for i, group in groups.items():
    mistake = 0
    total = len(group)
    batch = []
    for j, example in enumerate(tqdm(group)):
        if j == 1:
            break
        img_path = f"{data_path}/test/imgs/{set_['img_id'].iloc[example]}.jpg"
        _, ax = plt.subplots(1, 1, dpi=96)
        img_before = Image.open(img_path).convert("RGB")
        img_train = trans_train(img_before.copy())
        img_val = trans_val(img_before.copy())
        # _, axs = plt.subplots(1, 3, dpi=96)
        img_before.save(f"{output_path}/{i}_{j}_before.png")
        img_train.save(f"{output_path}/{i}_{j}_train.png")
        img_val.save(f"{output_path}/{i}_{j}_val.png")