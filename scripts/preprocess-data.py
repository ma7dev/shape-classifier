import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os
from tqdm import tqdm
# img_id, class_num
data_path = "/scratch/alotaima/datasets/v6-tmp"
output_path = "/scratch/alotaima/datasets/v6-shape-wall"
seq_len = {'COL': 2, 'GRV': 8, 'OP4': 2, 'SC': 2, 'STC4': 2}
train_seqs = []
for i in range(1,51):
    for seq_name, llen in seq_len.items():
        for j in range(1,llen+1):
            train_seqs.append(f"{seq_name}_00{str(i).zfill(2)}_0{j}")
val_seqs = []
for i in range(51,76):
    for seq_name, llen in seq_len.items():
        for j in range(1,llen+1):
            val_seqs.append(f"{seq_name}_00{str(i).zfill(2)}_0{j}")
set_names = ['frame_num', 'object_id', 'class_num', 'vis', 'xmin', 'ymin', 'width', 'height', 'conf']
padding = 5
if not os.path.exists(f'{output_path}'):
    os.makedirs(f'{output_path}')
if not os.path.exists(f'{output_path}/train'):
    os.makedirs(f'{output_path}/train')
if not os.path.exists(f'{output_path}/train/imgs'):
    os.makedirs(f'{output_path}/train/imgs')
if not os.path.exists(f'{output_path}/test'):
    os.makedirs(f'{output_path}/test')
if not os.path.exists(f'{output_path}/test/imgs'):
    os.makedirs(f'{output_path}/test/imgs')
classes_ = {
    0: {'name': 'cube', 'num': 0}, 
    1: {'name': 'cone', 'num': 1},
    2: {'name': 'circle frustum', 'num': 2},
    3: {'name': 'cylinder', 'num': 3},
    4: {'name': 'pyramid', 'num': 4},
    5: {'name': 'square frustum', 'num': 5},
    6: {'name': 'letter l', 'num': 6},
    7: {'name': 'skipped_1', 'num': -1}, # remove
    8: {'name': 'triangular prism', 'num': 7},
    9: {'name': 'car', 'num': 8},
    10: {'name': 'duck', 'num': 9},
    11: {'name': 'skipped_2', 'num': -1}, # remove
    12: {'name': 'sphere', 'num': 10},
    13: {'name': 'train', 'num': 11},
    14: {'name': 'trolley', 'num': 12},
    15: {'name': 'tube narrow', 'num': 13},
    16: {'name': 'tube wide', 'num': 14},
    17: {'name': 'turtle', 'num': 15},
    18: {'name': 'occluder_pole', 'num': 16}, # rename
    19: {'name': 'occluder_wall', 'num': 17}, # remove
}
missing = []
print('\ntrain')
filename = f"{output_path}/train/gt.txt"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile) 
    for train_seq in tqdm(train_seqs):
        if not os.path.exists(f'{data_path}/{train_seq}'):
            print('.',end="")
            missing.append(train_seq)
            continue
        df = pd.read_csv(f"{data_path}/{train_seq}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
        for img_num in df['frame_num'].unique():
            im = Image.open(f"{data_path}/{train_seq}/RGB/{str(img_num).zfill(6)}.png")
            objs = df[df['frame_num'] == img_num].values.tolist()
            for i, obj in enumerate(objs):
                # print(obj[0],obj[2],obj[4],obj[5],obj[6],obj[7])
                class_num = classes_[int(float(obj[2]))]['num']
                if class_num == -1:
                    continue
                img_width, img_height = im.size
                left = max(0.0, int(float(obj[4])) - padding)
                top = max(0.0, int(float(obj[5])) - padding)
                right = min(img_width, int(float(obj[4])) + int(float(obj[6])) + padding)
                bottom = min(img_height, int(float(obj[5])) + int(float(obj[7])) + padding)
                vis = float(obj[3])
                im1 = im.crop((left, top, right, bottom))
                example_num = f'{train_seq}_{img_num}_{i}'
                im1.save(f"{output_path}/train/imgs/{example_num}.jpg")
                csvwriter.writerow([example_num, vis, class_num])
print('\ntest')
filename = f"{output_path}/test/gt.txt"
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile) 
    for val_seq in tqdm(val_seqs):
        if not os.path.exists(f'{data_path}/{val_seq}'):
            print('.',end="")
            missing.append(val_seq)
            continue
        df = pd.read_csv(f"{data_path}/{val_seq}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
        for img_num in df['frame_num'].unique():
            im = Image.open(f"{data_path}/{val_seq}/RGB/{str(img_num).zfill(6)}.png")
            objs = df[df['frame_num'] == img_num].values.tolist()
            for i, obj in enumerate(objs):
                # print(obj[0],obj[2],obj[4],obj[5],obj[6],obj[7])
                class_num = classes_[int(float(obj[2]))]['num']
                if class_num == -1:
                    continue
                img_width, img_height = im.size
                left = max(0.0, int(float(obj[4])) - padding)
                top = max(0.0, int(float(obj[5])) - padding)
                right = min(img_width, int(float(obj[4])) + int(float(obj[6])) + padding)
                bottom = min(img_height, int(float(obj[5])) + int(float(obj[7])) + padding)
                vis = float(obj[3])
                im1 = im.crop((left, top, right, bottom))
                example_num = f'{val_seq}_{img_num}_{i}'
                im1.save(f"{output_path}/test/imgs/{example_num}.jpg")
                csvwriter.writerow([example_num, vis, class_num])

print(missing)
print(len(missing))