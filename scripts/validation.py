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
traceback.install()
pretty.install()
trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
data_path = "/scratch/alotaima/datasets/v6-shape"
set_names = ['img_id', 'vis', 'class_num']
classes = [
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
def _get_lowest(groups):
    min_ = math.inf
    for i in groups.keys():
        if min_ > len(groups[i]):
            min_ = len(groups[i])
    return min_
batch_size = 256
best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/7NV5M_main_256_0.001_10/epoch=9-step=189.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.eval()
model.to(device)

df = pd.read_csv(f"{data_path}/train/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
print(df['vis'].count())
# print(df['class_num'].unique())
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
# min_ = get_lowest(groups)
# inspect(groups)
lowest = _get_lowest(groups)
# selected = [random.sample(list(group), k=lowest) for _, group in groups.items()]
selected =  [list(group) for _, group in groups.items()]
iter_num = math.floor((len(groups.keys())*lowest) / batch_size)
# best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/best/31LSN_main_256_0.001_10/epoch=9-step=169.ckpt'
# print(selected)
# train
for i, group in enumerate(selected):
    mistake = 0
    total = len(group)
    batch = []
    for j, example in enumerate(group):
        img_path = f"{data_path}/train/imgs/{set_['img_id'].iloc[example]}.jpg"
        img = trans(Image.open(img_path).convert("RGB"))
        batch.append(img)
        if len(batch) == batch_size:
            preds = model.model(torch.stack(batch).to(device))
            preds = torch.argmax(preds, dim=1)
            for pred in preds:
                if pred != i:
                    mistake += 1
            batch = []
        elif (j+1) == total:
            if len(batch) > 1:
                preds = model.model(torch.stack(batch).to(device))
                preds = torch.argmax(preds, dim=1)
                for pred in preds:
                    if pred != i:
                        mistake += 1
                batch = []
            else:
                pred = model.model(img.to(device))
                pred = torch.argmax(pred, dim=1).item()
                if pred != i:
                    mistake += 1
    print(f"{classes[i]} ({i}): {len(group)} acc: {(total-mistake)/(total)}")
# test
df = pd.read_csv(f"{data_path}/test/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
print(df['vis'].count())
# print(df['class_num'].unique())
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
lowest = _get_lowest(groups)
# selected = [random.sample(list(group), k=lowest) for _, group in groups.items()]
selected =  [list(group) for _, group in groups.items()]
iter_num = math.floor((len(groups.keys())*lowest) / batch_size)
for i, group in groups.items():
    mistake = 0
    total = len(group)
    batch = []
    for j, example in enumerate(group):
        img_path = f"{data_path}/test/imgs/{set_['img_id'].iloc[example]}.jpg"
        img = trans(Image.open(img_path).convert("RGB"))
        batch.append(img)
        if len(batch) == batch_size:
            preds = model.model(torch.stack(batch).to(device))
            preds = torch.argmax(preds, dim=1)
            for pred in preds:
                if pred != i:
                    mistake += 1
            batch = []
        elif (j+1) == total:
            if len(batch) > 1:
                preds = model.model(torch.stack(batch).to(device))
                preds = torch.argmax(preds, dim=1)
                for pred in preds:
                    if pred != i:
                        mistake += 1
                batch = []
            else:
                pred = model.model(img.to(device))
                pred = torch.argmax(pred, dim=1).item()
                if pred != i:
                    mistake += 1
    print(f"{classes[i]} ({i}): {len(group)} acc: {(total-mistake)/(total)}")
# def flatten(t):
#     return [item for sublist in t for item in sublist]
# print(len(flatten([group for _, group in groups.items()])))
# print(groups.items())
# print(flatten([random.sample(list(group), k=2) for _, group in groups.items()]))
# print(len(df['img_id']))
# print(set_.iloc[0])
# print(set_['img_id'].iloc[0])
# print(set_.size)
# min_ = math.inf
# inspect(df.groups)
# for i in df.groups.keys():
#     if min_ > len(df.groups[i]):
#         min_ = len(df.groups[i])
    # print(f"{i}: {len(df.groups[i])}")

# print(min_)

# df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
#                               'Parrot', 'Parrot'],
#                    'Max Speed': [380., 370., 24., 26.]})
# print(max(flatten([group for group in groups.values()])))
# print(max(flatten([group for group in groups.values()])))