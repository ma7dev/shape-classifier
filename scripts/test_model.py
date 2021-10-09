import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import csv
import os
from tqdm import tqdm
from shape_classifier.pl.lightning_module import LitModel
from torchvision import transforms
import numpy as np

DATA_PATH = "/scratch/alotaima/datasets/v6/tracker-test"
OUTPUT_PATH = "/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs"
PADDING = 5
set_names = ['frame_num', 'object_id', 'class_num','vis', 'xmin', 'ymin', 'width', 'height', 'conf']
# df = pd.read_csv(f"{DATA_PATH}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
# print(df)
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
stroke_color = (0, 0, 255)
fill_color = (255, 123, 125)
rgb = np.random.randint(0,255,(100000, 3))
trans = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/best/31LSN_main_256_0.001_10/epoch=9-step=169.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
model.eval()
train_seq = "COL_0051_02"
df = pd.read_csv(f"{DATA_PATH}/{train_seq}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
for img_num in df['frame_num'].unique():
    im = Image.open(f"{DATA_PATH}/{train_seq}/RGB/{str(img_num).zfill(6)}.png").convert("RGB")
    draw = ImageDraw.Draw(im)
    objs = df[df['frame_num'] == img_num].values.tolist()
    for i, obj in enumerate(objs):
        # print(obj[0],obj[2],obj[4],obj[5],obj[6],obj[7])
        # frame_num 0
        # object_id 1
        # xmin, ymin, width, height, 2,3,4,5
        object_id = int(float(obj[1]))
        xmin = int(float(obj[2]))
        ymin = int(float(obj[3]))
        width = int(float(obj[4]))
        height = int(float(obj[5]))
        left = xmin - PADDING
        top = ymin - PADDING
        right = xmin + width + PADDING
        bottom = ymin + height + PADDING
        inp = trans(im.crop((left, top, right, bottom))).unsqueeze(0)
        preds = model.model(inp)
        preds = torch.argmax(preds, dim=1).item()
        print(preds, classes[preds])
        preds = classes[preds]
        draw.rectangle([xmin, ymin, xmin + width, ymin + height], outline=tuple(rgb[object_id]))
        draw.text((xmin,ymin), preds, fill=fill_color, stroke_width=7, stroke_fill=stroke_color)
        # csvwriter.writerow([example_num, vis, class_num])
    im.save(f"{OUTPUT_PATH}/test/imgs/{str(img_num).zfill(5)}.jpg")

# best_file = torch.load(best_file_path)
# print(best_file['state_dict'].keys())
# model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
# model.eval()
# model.load_state_dict(best_file)