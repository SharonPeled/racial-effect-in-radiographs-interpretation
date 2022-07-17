import os
import numpy as np
from CheXpert.generic_dataset import CheXpertDataset
from CheXpert.disease_prediction.utils import Configs


class CheXpertDiseaseDataset(CheXpertDataset):
    def __init__(self, data_dir, labels_filename, transform=None, target_transform=None):
        super().__init__(data_dir, labels_filename, transform, target_transform)
        self.transform_labels()

    def transform_labels(self):
        self.df_labels = self.original_labels.copy()
        self.df_labels.rename(columns={"Path": "original_path"}, inplace=True)
        self.df_labels['img_path'] = self.df_labels.original_path.apply(lambda p: os.path.abspath(os.path.join(self.data_dir, p[20:])))
        self.df_labels[Configs.UONES_COLUMNS] = self.df_labels[Configs.UONES_COLUMNS].abs()
        self.df_labels[Configs.UZEROS_COLUMNS] = self.df_labels[Configs.UZEROS_COLUMNS].applymap(lambda v: max(v, 0))
        self.df_labels[Configs.ANNOTATIONS_COLUMNS] = self.df_labels[Configs.ANNOTATIONS_COLUMNS].fillna(0).astype(np.float32)
        self.df_labels['patient_id'] = self.df_labels.original_path.apply(lambda p: p.split("/")[2])
        self.df_labels['study'] = self.df_labels.original_path.apply(lambda p: p.split("/")[3])
        self.df_labels['view'] = self.df_labels.original_path.apply(lambda p: p.split("/")[4])
        self.ann_cols = Configs.ANNOTATIONS_COLUMNS
