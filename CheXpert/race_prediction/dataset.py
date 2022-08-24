import pandas as pd
import os
from collections import defaultdict
from generic_dataset import GenericDataset
from CheXpert.race_prediction.utils import Configs


class CheXpertRaceDataset(GenericDataset):
    def __init__(self, data_dir, demo_filename, labels_filename, transform=None, target_transform=None, label_transform=True):
        super().__init__(data_dir, labels_filename, transform, target_transform)
        self.original_demo = pd.read_csv(os.path.join(self.data_dir, demo_filename))
        self.ann_cols = Configs.ANNOTATIONS_COLUMNS
        self.df_labels = self.original_labels.copy()
        if label_transform:
            self.transform_df_labels()

    def transform_df_labels(self):
        self.original_labels['img_path'] = self.original_labels.Path.apply(
            lambda p: os.path.abspath(os.path.join(self.data_dir, p[20:])))
        self.original_labels['PATIENT'] = self.original_labels.Path.apply(lambda p: p.split("/")[2])
        self.df_labels = CheXpertRaceDataset.generate_race_dummies(self.original_demo, 'PRIMARY_RACE', Configs.RACE_DICT)
        self.df_labels = self.df_labels.merge(self.original_labels, how='inner', on='PATIENT')

    @staticmethod
    def generate_race_dummies(df_labels, race_col, race_dict):
        reversed_race_dict = defaultdict(lambda: None,
                                         dict((v, k) for k, v_list in race_dict.items() for v in v_list))
        df_labels['race'] = df_labels[race_col].apply(lambda r: reversed_race_dict[r])
        df_labels.dropna(subset=['race'], inplace=True)
        race_encoding = pd.get_dummies(df_labels.race)
        df_labels[race_encoding.columns] = race_encoding
        return df_labels