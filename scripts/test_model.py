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
import pickle
import glob, os

DATA_TRACKING_PATH = "/scratch/alotaima/datasets/tracking-level2"
DATA_IMG_PATH = "/scratch/alotaima/datasets/level1"
OUTPUT_PATH = "/scratch/alotaima/outputs"
PADDING = 5
set_names = ['frame_num', 'object_id', 'xmin', 'ymin', 'width', 'height']
# df = pd.read_csv(f"{DATA_PATH}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
# print(df)
classes = [
    "cube",
    "pole",
    "else"
]
def most_frequent(List):
    return max(set(List), key = List.count)
stroke_color = (0, 0, 255)
fill_color = (255, 123, 125)
rgb = np.random.randint(0,255,(100000, 3))
trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])
# best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/best/31LSN_main_256_0.001_10/epoch=9-step=169.ckpt'
best_file_path = '/nfs/hpc/share/alotaima/mcs/shape-classifier/outputs/checkpoints/GYKCP_main_256_0.001_10/epoch=9-step=4979.ckpt'
    
model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
model.eval()
seqs = []
for file in glob.glob(f"{DATA_TRACKING_PATH}/*.txt"):
    file = file.replace(f"{DATA_TRACKING_PATH}/","").replace(".txt","")
    seqs.append(file)
for seq in tqdm(seqs):
    # print(seq)
    history= {}
    df = pd.read_csv(f"{DATA_TRACKING_PATH}/{seq}.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
    for img_num in df['frame_num'].unique():
        im = Image.open(f"{DATA_IMG_PATH}/{seq}/RGB/{str(img_num).zfill(6)}.png").convert("RGB")
        # draw = ImageDraw.Draw(im)
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
            # print(preds, classes[preds])
            preds_text = classes[preds]
            text = f"{preds_text} ({object_id})"
            # draw.rectangle([xmin, ymin, xmin + width, ymin + height], outline=tuple(rgb[object_id]))
            # draw.text((xmin,ymin), text, fill=fill_color, stroke_width=7, stroke_fill=stroke_color)
            if object_id not in history.keys():
                history[object_id] = {
                    "2d_position_history": {"t": [], "2dpos": []},
                    "shapes": [],
                    "shape": -1
                }
            history[object_id]["2d_position_history"]["t"].append(img_num)
            history[object_id]["2d_position_history"]["2dpos"].append([xmin,ymin,width, height])
            history[object_id]["shapes"].append(preds)
            # csvwriter.writerow([example_num, vis, class_num])
        if not os.path.isdir(f'{OUTPUT_PATH}/{seq}'):
            os.mkdir(f'{OUTPUT_PATH}/{seq}')
        # if not os.path.isdir(f'{OUTPUT_PATH}/{seq}/imgs'):
        #     os.mkdir(f'{OUTPUT_PATH}/{seq}/imgs')
        # im.save(f"{OUTPUT_PATH}/{seq}/imgs/{str(img_num).zfill(5)}.jpg")
    for object_id in history.keys():
        history[object_id]["shape"] = most_frequent(history[object_id]["shapes"])
        del history[object_id]["shapes"]
        # print(f"{object_id}: {history[object_id]['shape']}")
    with open(f"{OUTPUT_PATH}/{seq}/dict.pickle", 'wb') as f:
        pickle.dump(history, f)


# best_file = torch.load(best_file_path)
# print(best_file['state_dict'].keys())
# model = LitModel.load_from_checkpoint(checkpoint_path=best_file_path)
# model.eval()
# model.load_state_dict(best_file)