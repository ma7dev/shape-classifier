from typing import Optional, List, Tuple
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()
import torch
from torchvision.transforms import functional as F
import random
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from shape_classifier.data.datasets.mcs import MCS
from shape_classifier.data.samplers.simple_sampler import BatchSampler

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_transform(train: bool):
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.Resize((224,224)),             
            transforms.ToTensor(),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)], p=0.5),
            transforms.RandomApply([AddGaussianNoise(0., 0.1)], p=0.25),  
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([      
        transforms.ToTensor(),
        transforms.Resize((224,224)),     
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])

def collate_fn(batch: List[torch.Tensor]) -> Tuple[Tuple[torch.Tensor]]:
    """[summary]
    Args:
        batch (List[torch.Tensor]): [description]
    Returns:
        Tuple[Tuple[torch.Tensor]]: [description]
    """
    return tuple(zip(*batch))

class LitDataset(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str, 
        batch_size: int, 
        num_workers: int = 4    
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = -1 
        self.train_dataset = self.val_dataset = None
        self.train_sampler = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MCS(
                                f"{self.data_path}/train",
                                get_transform(train=False),
                                batch_size=self.batch_size,
                            )
            self.num_classes = self.train_dataset.num_classes
            self.val_dataset = MCS(
                                    f"{self.data_path}/test",
                                    get_transform(train=False),
                                    batch_size=self.batch_size,
                                )
            # self.train_sampler = BatchSampler(
            #     self.train_dataset,
            #     batch_size=self.batch_size,
            # )
            # self._train_dataloader = DataLoader(
            #     self.train_dataset, 
            #     batch_sampler=self.train_sampler,
            #     num_workers=self.num_workers,
            # )
            self._train_dataloader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
            self._val_dataloader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers
            )
        # if stage == "test" or stage is None:
        #     self.test_dataset = MCS(
        #                             f"{self.data_path}/test",
        #                             get_transform(train=False),
        #                             batch_size=self.batch_size,
        #                         )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset, 
    #         batch_size=self.batch_size, 
    #         num_workers=self.num_workers
    #     )