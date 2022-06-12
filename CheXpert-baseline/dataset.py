import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import *


class CheXpertDataset(Dataset):
    def __init__(self, mode, data_dir=r"..\data", transform=None, target_transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.original_labels = pd.read_csv(os.path.join(self.data_dir, "CheXpert-v1.0-small", f"{mode}.csv"))
        self.labels = self.original_labels.copy()
        self.transform_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_label = self.labels.loc[idx]
        image = Image.open(img_label.Path).convert("RGB")
        label = torch.from_numpy(img_label[Configs.ANNOTATIONS_COLUMNS].astype(float).values)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def transform_labels(self):
        self.labels.Path = self.labels.Path.apply(lambda p: os.path.join(self.data_dir, p))
        self.labels[Configs.UONES_COLUMNS] = self.labels[Configs.UONES_COLUMNS].abs()
        self.labels[Configs.UZEROS_COLUMNS] = self.labels[Configs.UZEROS_COLUMNS].applymap(lambda v: max(v, 0))
        self.labels[Configs.ANNOTATIONS_COLUMNS] = self.labels[Configs.ANNOTATIONS_COLUMNS].fillna(0).astype(float)
        self.labels['patient_id'] = self.labels.Path.apply(lambda p: p.split("/")[2])
        self.labels['study'] = self.labels.Path.apply(lambda p: p.split("/")[3])
        self.labels['view'] = self.labels.Path.apply(lambda p: p.split("/")[4])
