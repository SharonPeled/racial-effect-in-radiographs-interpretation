import os
import numpy as np
from generic_dataset import GenericDataset
from CheXpert.disease_prediction.utils import Configs


class CheXpertDiseaseDataset(GenericDataset):
    def __init__(self, data_dir, labels_filename, transform=None, target_transform=None, sample_weight_factor=None):
        super().__init__(data_dir, labels_filename, transform, target_transform)
        self.sample_weight_factor = sample_weight_factor
        self.transform_labels()

    def transform_labels(self):
        self.df_labels = self.original_labels.copy()
        self.df_labels.rename(columns={"Path": "original_path"}, inplace=True)
        self.df_labels['img_path'] = self.df_labels.original_path.apply(lambda p: os.path.abspath(os.path.join(self.data_dir, p[20:])))
        self.df_labels = CheXpertDiseaseDataset.impute_uncertainties(self.df_labels)
        self.df_labels['patient_id'] = self.df_labels.original_path.apply(lambda p: p.split("/")[2])
        self.df_labels['study'] = self.df_labels.original_path.apply(lambda p: p.split("/")[3])
        self.df_labels['view'] = self.df_labels.original_path.apply(lambda p: p.split("/")[4])
        self.ann_cols = Configs.ANNOTATIONS_COLUMNS
        if self.sample_weight_factor is not None:
            num_uncertainties = (self.original_labels[self.ann_cols]==-1).sum(axis=1)
            self.df_labels['weight'] = self.sample_weight_factor**num_uncertainties

    @staticmethod
    def impute_uncertainties(df_labels):
        df_labels[Configs.UONES_COLUMNS] = df_labels[Configs.UONES_COLUMNS].abs()
        df_labels[Configs.UZEROS_COLUMNS] = df_labels[Configs.UZEROS_COLUMNS].applymap(lambda v: max(v, 0))
        df_labels[Configs.ANNOTATIONS_COLUMNS] = df_labels[Configs.ANNOTATIONS_COLUMNS].fillna(0)
        df_labels[Configs.ANNOTATIONS_COLUMNS] =df_labels[Configs.ANNOTATIONS_COLUMNS].astype(np.float32)
        return df_labels
