"""Machine Common Sense Dataset definition"""
from typing import List, Dict, Any, Tuple
# credits: https://github.com/phil-bergmann/tracking_wo_bnw/
# blob/master/src/faster_rcnn_fpn/mot_data.py
import csv
import os
import math

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class MCS(Dataset):
    """Machine Common Sense Dataset

    Args:
        root (str): [description]
        transforms Compose, optional): [description]. Defaults to None.
        vis_threshold (float, optional): [description]. Defaults to 0.25.
        split_seqs (List, optional): [description]. Defaults to None.
        batch_size (int, optional): [description]. Defaults to 6.
        test (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self, root: str, 
        transforms = None, 
        vis_threshold: float = 0.25,
        batch_size: int = 6
    ):
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        # self.classes = [
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
        self.classes = [
            "cube",
            "pole",
            "else"
        ]
        self._img_paths = []
        self._batch_size = batch_size
        set_names = ['img_id', 'vis', 'class_num']
        df = pd.read_csv(f"{root}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
        # if "test" in root:
        #     print(root)
        #     self.set = ( 
        #         df
        #         [df['img_id'].str.contains("COL_0051_02", case=False)]
        #         [df['vis'] >= vis_threshold] # filter
        #         .reset_index() # reset indices
        #         # .sample(frac=1) # randomness
        #     )
        # else:
        self.set = ( 
            df
            .copy()
            [df['vis'] >= vis_threshold] # filter
            .reset_index() # reset indices
            .sample(frac=1) # randomness
        )
        # self.groups = (
        #     self.set
        #     .copy()
        #     .groupby(['class_num']) # grouping
        #     .groups
        # )
        self.before_groups = (
            self.set
            .copy()
            .groupby(['class_num']) # grouping
            .groups
        )
        self.groups = {
            0: [],
            1: [],
            2: []
        }
        self.mapping = [0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,0]
        for i, group in self.before_groups.items():
            if i == 0:
                self.groups[0].extend(group)
            elif i > 0 and i < 16:
                self.groups[2].extend(group)
            elif i == 17:
                self.groups[2].extend(group)
            elif i == 16:
                self.groups[1].extend(group)
        # if "train" in root:
        #     print(root)
        #     self.lowest = self._get_lowest(self.groups)
        #     self.len = math.floor((len(self.groups.keys())*self.lowest))
        # else:
        #     self.len = len(self.set)
        self.len = len(self.set)

    def _get_lowest(self, groups):
        min_ = math.inf
        for i in groups.keys():
            if min_ > len(groups[i]):
                min_ = len(groups[i])
        return min_
    def __str__(self) -> str:
        return f"{self.root}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # print(len(self.set), idx)
        img_path = f"{self.root}/imgs/{self.set['img_id'].iloc[idx]}.jpg"
        img = Image.open(img_path).convert("RGB")
        target = self.mapping[self.set['class_num'].iloc[idx]]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    
    # def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # print(len(self.set), idx)
    #     img_path = f"{self.root}/imgs/{self.set['img_id'].iloc[idx]}.jpg"
    #     img = Image.open(img_path).convert("RGB")
    #     target = self.set['class_num'].iloc[idx]
    #     if self.transforms is not None:
    #         img = self.transforms(img)
    #     return img, target

    def __len__(self) -> int:
        return self.len

    @property
    def num_classes(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return len(self.classes)