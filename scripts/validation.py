import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
import csv
import os
from tqdm import tqdm
import math
data_path = "/scratch/alotaima/datasets/v6-shape"
set_names = ['img_id', 'vis', 'class_num']
df = pd.read_csv(f"{data_path}/test/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
print(df['vis'].count())
# print(df['class_num'].unique())
df = (
    df
    [df['vis'] > 0.5]
    .groupby('class_num')
    .apply(lambda x: x.sample(n=269))
    .reset_index(drop = True)
    .sample(frac=1)
)
print(df.to_dict().keys())
# min_ = math.inf
# for i in df.groups.keys():
#     if min_ > len(df.groups[i]):
#         min_ = len(df.groups[i])
    # print(f"{i}: {len(df.groups[i])}")

# print(min_)