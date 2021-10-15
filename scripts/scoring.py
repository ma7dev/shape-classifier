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
import pickle
import numpy as np
# from visdom import Visdom
# port = 8097
# server = "127.0.0.1"
# base_url = '/'
# username = ''
# password = ''
# vis = Visdom(port=port, server=server, base_url=base_url)
# assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'

np.set_printoptions(suppress=True)
def get_transform(train: bool):
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize((224,224)),
            # transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
# trans_train = get_transform(True)
trans_val = get_transform(False)
data_path = "/scratch/alotaima/datasets/v6-shape"
output_path ="/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/vis/aug"
set_names = ['img_id', 'vis', 'class_num']
# old_classes = [
#     'cube',
#     'cone',
#     'circle frustum',
#     'cylinder',
#     'pyramid',
#     'square frustum',
#     'letter l',
#     'triangular prism',
#     'car',
#     'duck',
#     'sphere',
#     'train',
#     'trolley',
#     'tube narrow',
#     'tube wide',
#     'turtle',
#     'pole',
# ]
# classes = [
#     "cube",
#     "pole",
#     "else"
# ]
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
df_test = pd.read_csv(f"{data_path}/test/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
set_test = (
    df_test
    [df_test['vis'] >= 0.2] # filter
    .reset_index()
)
print(len(set_test['class_num']))
print(set_test.count())
groups_test = (
    set_test
    .groupby(['class_num']) # grouping
    # ['class_num']
    # .apply(list)
    # .to_dict()
    .groups
)
print(groups_test.keys())
print([len(group) for _, group in groups_test.items()])
# new_groups_test = {
#     0: [],
#     1: [],
#     2: []
# }
# for i, group in groups_test.items():
#     if i == 0:
#         new_groups_test[0].extend(group)
#     elif i > 0 and i < 16:
#         new_groups_test[2].extend(group)
#     elif i == 16:
#         new_groups_test[1].extend(group)
# print("test", len(new_groups_test[0]), len(new_groups_test[1]), len(new_groups_test[2]))
df_train = pd.read_csv(f"{data_path}/train/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
set_train = (
    df_train
    [df_train['vis'] >= 0.2] # filter
    .reset_index()
)
print(len(set_train['class_num']))
print(set_train.count())
groups_train = (
    set_train
    .groupby(['class_num']) # grouping
    # ['class_num']
    # .apply(list)
    # .to_dict()
    .groups
)
print(groups_train.keys())
print([len(group) for _, group in groups_train.items()])
# new_groups_train = {
#     0: [],
#     1: [],
#     2: []
# }
# for i, group in groups_train.items():
#     if i == 0:
#         new_groups_train[0].extend(group)
#     elif i > 0 and i < 16:
#         new_groups_train[2].extend(group)
#     elif i == 16:
#         new_groups_train[1].extend(group)

# print("train", len(new_groups_train[0]), len(new_groups_train[1]), len(new_groups_train[2]))
batch_size = 256
# best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/7NV5M_main_256_0.001_10/epoch=9-step=189.ckpt'
# best_file_path = "/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/5M5NI_random_sampler_256_0.001_10/epoch=6-step=3485.ckpt"
# best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/LS1AK_main_256_0.001_10/epoch=9-step=4979.ckpt'
# best 3
best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/GYKCP_main_256_0.001_10/epoch=9-step=4979.ckpt'
# best 17
# best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/UNLMT_main_256_0.001_10/epoch=9-step=4979.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(model)
print(device)
model.eval()
model.to(device)
mistakes_train = []
mistakes_cat_train = np.zeros((len(groups_train.keys())))
total_cat_train = np.zeros((len(groups_train.keys())))
matrix_train = np.zeros((len(groups_train.keys()),len(groups_train.keys())))
print("--------------train")
for i, group in groups_train.items():
    mistake = []
    total = len(group)
    batch = []
    idx_batch = []
    # target = 0
    # if i == 16:
    #     target = 1
    # elif i > 0 and i < 16:
    #     target = 2
    target = i
    total_cat_train[target] += len(group)
    for j, example in enumerate(group):
        img_path = f"{data_path}/train/imgs/{set_train['img_id'].iloc[example]}.jpg"
        img = trans_val(Image.open(img_path).convert("RGB"))
        idx_batch.append(example)
        batch.append(img)
        if len(batch) == batch_size:
            preds = model.model(torch.stack(batch).to(device))
            preds = torch.argmax(preds, dim=1)
            for k, pred in enumerate(preds):
                pred = pred.item()
                matrix_train[pred][target] +=1
                # tmp_pred = pred
                if pred != target:
                #     if target == 0 or target == 1:
                #         matrix_train[pred][target] +=1
                #     else:
                #         matrix_train[pred][i+1] +=1 
                    mistake.append(idx_batch[k])
                    mistakes_cat_train[target] += 1
                # else:
                #     if target == 0 or target == 1:
                #         matrix_train[pred][target] += 1
                #     else:
                #         matrix_train[pred][-1] += 1
            batch = []
            idx_batch = []
        elif (j+1) == total:
            if len(batch) > 1:
                preds = model.model(torch.stack(batch).to(device))
                preds = torch.argmax(preds, dim=1)
                for k, pred in enumerate(preds):
                    pred = pred.item()
                    matrix_train[pred][target] +=1
                    # tmp_pred = pred
                    if pred != target:
                    #     if target == 0 or target == 1:
                    #         matrix_train[pred][target] +=1
                    #     else:
                    #         matrix_train[pred][i+1] +=1 
                        mistake.append(idx_batch[k])
                        mistakes_cat_train[target] += 1
                    # else:
                    #     if target == 0 or target == 1:
                    #         matrix_train[pred][target] += 1
                    #     else:
                    #         matrix_train[pred][-1] += 1
                batch = []
                idx_batch = []
            else:
                pred = model.model(img.to(device))
                pred = torch.argmax(pred, dim=1).item()
                matrix_train[pred][target] +=1
                # tmp_pred = pred
                if pred != target:
                #     if target == 0 or target == 1:
                #         matrix_train[pred][target] +=1
                #     else:
                #         matrix_train[pred][i+1] +=1 
                    mistake.append(idx_batch[0])
                    mistakes_cat_train[target] += 1
                # else:
                #     if target == 0 or target == 1:
                #         matrix_train[pred][target] += 1
                #     else:
                #         matrix_train[pred][-1] += 1
    print(f"{classes[i]} ({i}, {target}): {len(group)} acc: {(total-len(mistake))/(total)}")
    mistakes_train.extend(mistake)
print("--------------train-3")
for i, _ in enumerate(total_cat_train):
    total = total_cat_train[i]
    mistake = mistakes_cat_train[i]
    print(f"{classes[i]} ({i}): {total} acc: {(total-mistake)/(total)}")
print(matrix_train)
print("--------------test")
mistakes_test = []
mistakes_cat_test = np.zeros((len(groups_test.keys())))
total_cat_test = np.zeros((len(groups_test.keys())))
matrix_test = np.zeros((len(groups_test.keys()),len(groups_test.keys())))
for i, group in groups_test.items():
    mistake = []
    total = len(group)
    batch = []
    idx_batch = []
    # target = 0
    # if i == 16:
    #     target = 1
    # elif i > 0 and i < 16:
    #     target = 2
    target = i
    total_cat_test[target] += len(group)
    for j, example in enumerate(group):
        img_path = f"{data_path}/test/imgs/{set_test['img_id'].iloc[example]}.jpg"
        img = trans_val(Image.open(img_path).convert("RGB"))
        idx_batch.append(example)
        batch.append(img)
        if len(batch) == batch_size:
            preds = model.model(torch.stack(batch).to(device))
            preds = torch.argmax(preds, dim=1)
            for k, pred in enumerate(preds):
                pred = pred.item()
                matrix_test[pred][target] += 1
                if pred != target:
                #     if target == 0 or target == 1:
                #         matrix_test[pred][target] +=1
                #     else:
                #         matrix_test[pred][i+1] +=1 
                    mistake.append(idx_batch[k])
                    mistakes_cat_test[target] += 1
                # else:
                #     if target == 0 or target == 1:
                #         matrix_test[pred][target] += 1
                #     else:
                #         matrix_test[pred][-1] += 1
            batch = []
            idx_batch = []
        elif (j+1) == total:
            if len(batch) > 1:
                preds = model.model(torch.stack(batch).to(device))
                preds = torch.argmax(preds, dim=1)
                for k, pred in enumerate(preds):
                    pred = pred.item()
                    matrix_test[pred][target] += 1
                    if pred != target:
                        # if target == 0 or target == 1:
                        #     matrix_test[pred][target] +=1
                        # else:
                        #     matrix_test[pred][i+1] +=1 
                        mistake.append(idx_batch[k])
                        mistakes_cat_test[target] += 1
                    # else:
                    #     if target == 0 or target == 1:
                    #         matrix_test[pred][target] += 1
                    #     else:
                    #         matrix_test[pred][-1] += 1
                batch = []
                idx_batch = []
            else:
                pred = model.model(img.to(device))
                pred = torch.argmax(pred, dim=1).item()
                matrix_test[pred][target] += 1
                if pred != target:
                    # if target == 0 or target == 1:
                    #     matrix_test[pred][target] +=1
                    # else:
                    #     matrix_test[pred][i+1] +=1 
                    mistake.append(idx_batch[0])
                    mistakes_cat_test[target] += 1
                # else:
                #     if target == 0 or target == 1:
                #         matrix_test[pred][target] += 1
                #     else:
                #         matrix_test[pred][-1] += 1
    print(f"{classes[i]} ({i}, {target}): {len(group)} acc: {(total-len(mistake))/(total)}")
    mistakes_test.extend(mistake)
print("--------------test-3")
for i, _ in enumerate(total_cat_test):
    total = total_cat_test[i]
    mistake = mistakes_cat_test[i]
    print(f"{classes[i]} ({i}): {total} acc: {(total-mistake)/(total)}")
print(matrix_test)
# with open('/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/analysis/mistakes_train.pkl', 'wb') as file:
#     pickle.dump(mistakes_train, file)
# with open('/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/analysis/mistakes_test.pkl', 'wb') as file:
#     pickle.dump(mistakes_test, file)