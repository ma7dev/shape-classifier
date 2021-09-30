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

def get_transform(train: bool):
    # if train:
    #     transforms.append(RandomHorizontalFlip(0.5))
    if train:
        return transforms.Compose([
            transforms.RandomRotation(10),      # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(224),             # resize shortest side to 224 pixels
            transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
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

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MCS(
                                f"{self.data_path}/train",
                                get_transform(train=False),
                                batch_size=self.batch_size,
                                # test=True
                            )
            self.num_classes = self.train_dataset.num_classes
            self.val_dataset = MCS(
                                    f"{self.data_path}/train",
                                    get_transform(train=False),
                                    batch_size=self.batch_size,
                                    # test=True
                                )

        # if stage == "test" or stage is None:
        #     self.dataset_test = CIFAR10(
        #         self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.dataset_test, 
    #         batch_size=self.batch_size, 
    #         num_workers=self.num_workers
    #     )