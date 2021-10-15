"""Batch sampler class definition"""
from typing import List
import math
import random

from torch.utils import data


class BatchSampler(data.sampler.Sampler):
    """Batch sampler class

    Args:
        dataset (data.Dataset): [description]
        batch_size (int): [description]
        drop_last (bool, optional): [description]. Defaults to True.
        shuffle (bool, optional): [description]. Defaults to True.
    """

    def __init__(
        self, 
        dataset: data.Dataset, 
        batch_size: int
    ):
        super(BatchSampler, self).__init__(dataset)
        self.batch_size = batch_size
        self.dataset = dataset
        self.lowest = self._get_lowest(dataset.groups)
        self.iter_num = self._length()
        self.selected = {}
        self.examples = []

    def _length(self) -> int:
        return math.floor((len(self.dataset.groups.keys())*self.lowest) / self.batch_size)

    def _get_lowest(self, groups):
        min_ = math.inf
        for i in groups.keys():
            if min_ > len(groups[i]):
                min_ = len(groups[i])
        return min_
    def _flatten(self,t):
        return [item for sublist in t for item in sublist]
    def _build_sequecnes(self):
        selected = [random.sample(list(group), k=self.lowest) for _, group in self.dataset.groups.items()]
        return self._flatten(selected)

    def __iter__(self) -> List[int]:
        # build seqs
        self.examples = self._build_sequecnes()
        random.shuffle(self.examples)
        # batch
        for i in range(self.iter_num):
            batch = self.examples[i*self.batch_size:(i+1)*self.batch_size]
            # print(batch)
            yield batch

    def __len__(self) -> int:
        return self.iter_num