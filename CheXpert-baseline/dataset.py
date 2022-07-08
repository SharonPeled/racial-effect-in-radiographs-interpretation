import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit


class CheXpertDataset(Dataset):
    def __init__(self, labels_filename, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.labels_filename = labels_filename
        self.transform = transform
        self.target_transform = target_transform
        self.original_labels = pd.read_csv(os.path.join(self.data_dir, labels_filename))
        self.labels = self.original_labels.copy()
        self.transform_labels()

    @classmethod
    def _create_dataset(cls, other_dataset, labels_filename):
        return cls(labels_filename, other_dataset.data_dir, other_dataset.transform, other_dataset.target_transform)

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
        self.labels.Path = self.labels.Path.apply(lambda p: os.path.abspath(os.path.join(self.data_dir, p[20:])))
        self.labels[Configs.UONES_COLUMNS] = self.labels[Configs.UONES_COLUMNS].abs()
        self.labels[Configs.UZEROS_COLUMNS] = self.labels[Configs.UZEROS_COLUMNS].applymap(lambda v: max(v, 0))
        self.labels[Configs.ALL_ANNOTATIONS_COLUMNS] = self.labels[Configs.ALL_ANNOTATIONS_COLUMNS].fillna(0).astype(np.float32)
        self.labels['patient_id'] = self.labels.original_path.apply(lambda p: p.split("/")[2])
        self.labels['study'] = self.labels.original_path.apply(lambda p: p.split("/")[3])
        self.labels['view'] = self.labels.original_path.apply(lambda p: p.split("/")[4])

    def train_test_split(self, test_size):
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=Configs.SEED)
        train_inds, test_inds = next(splitter.split(self.labels, groups=self.labels['patient_id']))
        train = self.original_labels.iloc[train_inds]
        test = self.original_labels.iloc[test_inds]
        train_filename = f"train_split_{int(100*(1-test_size))}.csv"
        test_filename = f"train_split_{int(100*test_size)}.csv"
        train.to_csv(os.path.join(self.data_dir, train_filename), index=False)
        test.to_csv(os.path.join(self.data_dir, test_filename), index=False)
        return CheXpertDataset._create_dataset(self, train_filename), CheXpertDataset._create_dataset(self, test_filename)



