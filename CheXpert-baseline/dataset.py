import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import *
from sklearn.model_selection import train_test_split

class CheXpertDataset(Dataset):
    def __init__(self, mode, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.original_labels = pd.read_csv(os.path.join(self.data_dir, "CheXpert-v1.0-small", f"{mode}.csv"))
        self.labels = self.original_labels.copy()
        self.transform_labels()

    @classmethod
    def _create_dataset(cls, other_dataset, labels):
        cls.data_dir = other_dataset.data_dir
        cls.mode = other_dataset.mode
        cls.target_transform = other_dataset.target_transform
        cls.original_labels = other_dataset.original_labels[other_dataset.original_labels.Path.isin(
            labels.original_path)]
        cls.labels = other_dataset.labels.copy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_label = self.labels.loc[idx]
        image = Image.open(img_label.Path).convert("RGB")
        label = torch.from_numpy(img_label[Configs.CHALLENGE_ANNOTATIONS_COLUMNS].astype(np.float32).values)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def transform_labels(self):
        self.labels['original_path'] = self.labels.Path
        self.labels.Path = self.labels.Path.apply(lambda p: os.path.join(self.data_dir, p))
        self.labels[Configs.UONES_COLUMNS] = self.labels[Configs.UONES_COLUMNS].abs()
        self.labels[Configs.UZEROS_COLUMNS] = self.labels[Configs.UZEROS_COLUMNS].applymap(lambda v: max(v, 0))
        self.labels[Configs.ALL_ANNOTATIONS_COLUMNS] = self.labels[Configs.ALL_ANNOTATIONS_COLUMNS].fillna(0).astype(np.float32)
        self.labels['patient_id'] = self.labels.Path.apply(lambda p: p.split("/")[2])
        self.labels['study'] = self.labels.Path.apply(lambda p: p.split("/")[3])
        self.labels['view'] = self.labels.Path.apply(lambda p: p.split("/")[4])

    def train_test_split(self, test_size):
        train_test_split(test_size=test_size, random_state=Configs.SEED, shuffle=True)


