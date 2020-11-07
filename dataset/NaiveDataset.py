import json
import os
from torch.utils.data import Dataset
import random

class NaiveDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode

        self.data_path = config.get("data", "%s_data_path" % mode)
        self.data = json.load(open(self.data_path, 'r'))
        if mode == 'train':
            self.data = [d for d in self.data if len(d['labels']) != 0]
            random.shuffle(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)