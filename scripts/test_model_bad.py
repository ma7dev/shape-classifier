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
import glob
traceback.install()
pretty.install()
trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
DATA_PATH = "/scratch/alotaima/datasets/level1"
OUTPUT_PATH = '/scratch/alotaima/'
PADDING = 5
set_names = ['frame_num', 'object_id', 'class_num', 'vis', 'xmin', 'ymin', 'width', 'height']
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
    'cube',
    'pole',
    'else'
]
batch_size = 256

best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/GYKCP_main_256_0.001_10/epoch=9-step=4979.ckpt'
    
model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.eval()
model.to(device)
results = {
    'seq': [],
    'mistakes': [],
    'total': [],
    'results': []
}
classes_ = {
    0: {'name': 'cube', 'num': 0}, 
    1: {'name': 'cone', 'num': 2},
    2: {'name': 'circle frustum', 'num': 2},
    3: {'name': 'cylinder', 'num': 2},
    4: {'name': 'pyramid', 'num': 2},
    5: {'name': 'square frustum', 'num': 2},
    6: {'name': 'letter l', 'num': 2},
    7: {'name': 'skipped_1', 'num': -1}, # remove
    8: {'name': 'triangular prism', 'num': 2},
    9: {'name': 'car', 'num': 2},
    10: {'name': 'duck', 'num': 2},
    11: {'name': 'skipped_2', 'num': -1}, # remove
    12: {'name': 'sphere', 'num': 2},
    13: {'name': 'train', 'num': 2},
    14: {'name': 'trolley', 'num': 2},
    15: {'name': 'tube narrow', 'num': 2},
    16: {'name': 'tube wide', 'num': 2},
    17: {'name': 'turtle', 'num': 2},
    18: {'name': 'occluder_pole', 'num': 1}, # rename
    19: {'name': 'occluder_wall', 'num': -1}, # remove
}
seqs = []
for file in glob.glob(f"{DATA_PATH}/*"):
    file = file.replace(f"{DATA_PATH}/","")
    seqs.append(file)
for seq in tqdm(seqs):
    df = pd.read_csv(f"{DATA_PATH}/{seq}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
    # print(df['class_num'].unique())
    set_ = (
        df
        [df['vis'] >= 0.25] # filter
        .reset_index()
    )
    groups = (
        set_
        .groupby(['class_num']) # grouping
        # ['class_num']
        # .apply(list)
        # .to_dict()
        .groups
    )
    new_groups = {}
    print(groups.keys())
    for i, group in groups.items():
        _id = classes_[i]['num']
        if _id != -1:
            if _id not in new_groups:
                new_groups[_id] = []
            new_groups[_id].extend(group)
    print(new_groups.keys())
    total = 0
    mistakes = 0
    for i, group in new_groups.items():
        total += len(group)
        batch = []
        for j, example in enumerate(group):
            img_path = f"{DATA_PATH}/{seq}/RGB/{str(set_['frame_num'].iloc[example]).zfill(6)}.png"
            object_id = int(float(set_['object_id'].iloc[example]))
            xmin = int(float(set_['xmin'].iloc[example]))
            ymin = int(float(set_['ymin'].iloc[example]))
            width = int(float(set_['width'].iloc[example]))
            height = int(float(set_['height'].iloc[example]))
            left = xmin - PADDING
            top = ymin - PADDING
            right = xmin + width + PADDING
            bottom = ymin + height + PADDING
            img = trans(Image.open(img_path).convert("RGB").crop((left, top, right, bottom)))
            batch.append(img)
            if len(batch) == batch_size:
                preds = model.model(torch.stack(batch).to(device))
                preds = torch.argmax(preds, dim=1)
                for pred in preds:
                    if pred != i:
                        mistakes += 1
                batch = []
            elif (j+1) == total:
                if len(batch) > 1:
                    preds = model.model(torch.stack(batch).to(device))
                    preds = torch.argmax(preds, dim=1)
                    for pred in preds:
                        if pred != i:
                            mistakes += 1
                    batch = []
                else:
                    pred = model.model(img.to(device))
                    pred = torch.argmax(pred, dim=1).item()
                    if pred != i:
                        mistakes += 1
    if mistakes > 0:
        print(seq, mistakes, total)
    results['seq'].append(seq)
    results['mistakes'].append(mistakes)
    results['total'].append(total)
    results['results'].append((total-mistakes)/total)
# # results = {
# #     'seq': [0,1,2],
# #     'results': [1,0.5,0.7]
# # }
df_sort = pd.DataFrame(results)
df_sort.to_pickle(f"{OUTPUT_PATH}/results.pkl")
print(df_sort.sort_values(by=['results'])[:10])
# # test
# df = pd.read_csv(f"{DATA_PATH}/test/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
# print(df['vis'].count())
# # print(df['class_num'].unique())
# set_ = (
#     df
#     [df['vis'] >= 0.25] # filter
#     .reset_index()
# )
# print(len(set_['class_num']))
# print(set_.count())
# groups = (
#     set_
#     .groupby(['class_num']) # grouping
#     # ['class_num']
#     # .apply(list)
#     # .to_dict()
#     .groups
# )
# lowest = _get_lowest(groups)
# # selected = [random.sample(list(group), k=lowest) for _, group in groups.items()]
# selected =  [list(group) for _, group in groups.items()]
# iter_num = math.floor((len(groups.keys())*lowest) / batch_size)
# for i, group in groups.items():
#     mistake = 0
#     total = len(group)
#     batch = []
#     for j, example in enumerate(group):
#         img_path = f"{DATA_PATH}/test/imgs/{set_['img_id'].iloc[example]}.jpg"
#         img = trans(Image.open(img_path).convert("RGB"))
#         batch.append(img)
#         if len(batch) == batch_size:
#             preds = model.model(torch.stack(batch).to(device))
#             preds = torch.argmax(preds, dim=1)
#             for pred in preds:
#                 if pred != i:
#                     mistake += 1
#             batch = []
#         elif (j+1) == total:
#             if len(batch) > 1:
#                 preds = model.model(torch.stack(batch).to(device))
#                 preds = torch.argmax(preds, dim=1)
#                 for pred in preds:
#                     if pred != i:
#                         mistake += 1
#                 batch = []
#             else:
#                 pred = model.model(img.to(device))
#                 pred = torch.argmax(pred, dim=1).item()
#                 if pred != i:
#                     mistake += 1
#     print(f"{classes[i]} ({i}): {len(group)} acc: {(total-mistake)/(total)}")
# # def flatten(t):
# #     return [item for sublist in t for item in sublist]
# # print(len(flatten([group for _, group in groups.items()])))
# # print(groups.items())
# # print(flatten([random.sample(list(group), k=2) for _, group in groups.items()]))
# # print(len(df['img_id']))
# # print(set_.iloc[0])
# # print(set_['img_id'].iloc[0])
# # print(set_.size)
# # min_ = math.inf
# # inspect(df.groups)
# # for i in df.groups.keys():
# #     if min_ > len(df.groups[i]):
# #         min_ = len(df.groups[i])
#     # print(f"{i}: {len(df.groups[i])}")

# # print(min_)

# # df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
# #                               'Parrot', 'Parrot'],
# #                    'Max Speed': [380., 370., 24., 26.]})
# # print(max(flatten([group for group in groups.values()])))
# # print(max(flatten([group for group in groups.values()])))