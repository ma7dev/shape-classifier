import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os
from tqdm import tqdm
# img_id, class_num
data_path = "/scratch/alotaima/datasets/v6-tmp"
output_path = "/scratch/alotaima/datasets/v6-shape"
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
if not os.path.exists(f'{output_path}/train'):
    os.makedirs(f'{output_path}/train')
if not os.path.exists(f'{output_path}/train/imgs'):
    os.makedirs(f'{output_path}/train/imgs')
if not os.path.exists(f'{output_path}/test'):
    os.makedirs(f'{output_path}/test')
if not os.path.exists(f'{output_path}/test/imgs'):
    os.makedirs(f'{output_path}/test/imgs')
missing = []
# print('\ntrain')
# filename = f"{output_path}/train/gt.txt"
# with open(filename, 'w') as csvfile:
#     csvwriter = csv.writer(csvfile) 
#     for train_seq in tqdm(train_seqs):
#         if not os.path.exists(f'{data_path}/{train_seq}'):
#             print('.',end="")
#             missing.append(train_seq)
#             continue
#         df = pd.read_csv(f"{data_path}/{train_seq}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
#         for img_num in df['frame_num'].unique():
#             im = Image.open(f"{data_path}/{train_seq}/RGB/{str(img_num).zfill(6)}.png")
#             objs = df[df['frame_num'] == img_num].values.tolist()
#             for i, obj in enumerate(objs):
#                 # print(obj[0],obj[2],obj[4],obj[5],obj[6],obj[7])
#                 class_num = int(float(obj[2]))
#                 left = int(float(obj[4])) - padding
#                 top = int(float(obj[5])) - padding
#                 right = int(float(obj[4])) + int(float(obj[6])) + padding
#                 bottom = int(float(obj[5])) + int(float(obj[7])) + padding
#                 vis = float(obj[3])
#                 im1 = im.crop((left, top, right, bottom))
#                 example_num = f'{train_seq}_{img_num}_{i}'
#                 im1.save(f"{output_path}/train/imgs/{example_num}.jpg")
#                 csvwriter.writerow([example_num, vis, class_num])
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
                class_num = int(float(obj[2]))
                left = int(float(obj[4])) - padding
                top = int(float(obj[5])) - padding
                right = int(float(obj[4])) + int(float(obj[6])) + padding
                bottom = int(float(obj[5])) + int(float(obj[7])) + padding
                vis = float(obj[3])
                im1 = im.crop((left, top, right, bottom))
                example_num = f'{val_seq}_{img_num}_{i}'
                im1.save(f"{output_path}/test/imgs/{example_num}.jpg")
                csvwriter.writerow([example_num, vis, class_num])

print(missing)
print(len(missing))