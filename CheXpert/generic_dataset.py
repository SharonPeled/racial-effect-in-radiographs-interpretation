import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import os


class CheXpertDataset(Dataset):
    def __init__(self, data_dir, labels_filename, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.labels_filename = labels_filename
        self.original_labels = pd.read_csv(os.path.join(self.data_dir, labels_filename))
        self.df_labels = None
        self.ann_cols = None

    # @classmethod
    # def _create_dataset(cls, other_dataset, labels_filename):
    #     return cls(other_dataset.data_dir, labels_filename,
    #                other_dataset.transform, other_dataset.target_transform)

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

    @staticmethod
    def train_test_split(self, test_size, seed, df, group_col='patient_id'):
        splitter = GroupShuffleSplit(test_size=test_size, n_splits=2, random_state=seed)
        train_inds, test_inds = next(splitter.split(seed, groups=group_col))
        train = df.iloc[train_inds]
        test = df.iloc[test_inds]
        return train, test
        # train_filename = f"train_split_{int(100*(1-test_size))}.csv"
        # test_filename = f"train_split_{int(100*test_size)}.csv"
        # train.to_csv(os.path.join(self.data_dir, train_filename), index=False)
        # test.to_csv(os.path.join(self.data_dir, test_filename), index=False)
        # return CheXpertDataset._create_dataset(self, train_filename), CheXpertDataset._create_dataset(self, test_filename)




