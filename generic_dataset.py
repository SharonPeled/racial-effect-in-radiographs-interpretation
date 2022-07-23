import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class GenericDataset(Dataset):
    def __init__(self, data_dir, labels_filename, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.labels_filename = labels_filename
        self.original_labels = pd.read_csv(os.path.join(self.data_dir, labels_filename))
        self.df_labels = None
        self.ann_cols = None

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):
        example = self.df_labels.loc[idx]
        image = Image.open(example.img_path).convert("RGB")
        label = torch.from_numpy(example[self.ann_cols].astype(np.float32).values)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_attributes(self, columns):
        return self.df_labels[columns]





