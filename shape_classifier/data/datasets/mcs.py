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
        batch_size: int = 6,
        sample_size: int = 269,
    ):
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = [
            'cube',
            'cone',
            'circle frustum',
            'cylinder',
            'pyramid',
            'square frustum',
            'letter l',
            'skipped_1',
            'triangular prism',
            'car',
            'duck',
            'skipped_2',
            'sphere',
            'train',
            'trolley',
            'tube narrow',
            'tube wide',
            'turtle',
            'occluder_pole',
            'occluder_wall',
        ]
        self._img_paths = []
        self._batch_size = batch_size
        set_names = ['img_id', 'vis', 'class_num']
        df = pd.read_csv(f"{root}/gt.txt", usecols=range(0,len(set_names)), names=set_names, header=None)
        self.set = (
            df
            [df['vis'] > vis_threshold]
            .groupby('class_num')
            .apply(lambda x: x.sample(n=sample_size))
            .reset_index(drop = True)
            .to_dict()
        )
        
        

    def __str__(self) -> str:
        return f"{self.root}"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = f"{self.root}/imgs/{self.set['img_id'][idx]}.jpg"
        img = Image.open(img_path).convert("RGB")
        target = self.set['class_num'][idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self) -> int:
        return len(self.set['img_id'])

    @property
    def num_classes(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return len(self._classes)