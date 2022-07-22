import pandas as pd
import os
from collections import defaultdict
from generic_dataset import GenericDataset
from CheXpert.race_prediction.utils import Configs


class CheXpertRaceDataset(GenericDataset):
    def __init__(self, data_dir, demo_filename, labels_filename, transform=None, target_transform=None):
        super().__init__(data_dir, labels_filename, transform, target_transform)
        self.original_demo = pd.read_csv(os.path.join(self.data_dir, demo_filename))
        self.transform_df_labels()

    def transform_df_labels(self):
        self.df_labels = self.original_demo.copy()
        self.original_labels['img_path'] = self.original_labels.Path.apply(
            lambda p: os.path.abspath(os.path.join(self.data_dir, p[20:])))
        self.original_labels['PATIENT'] = self.original_labels.Path.apply(lambda p: p.split("/")[2])
        reversed_race_dict = defaultdict(lambda: None,
                                         dict((v, k) for k, v_list in Configs.RACE_DICT.items() for v in v_list))
        self.df_labels['race'] = self.df_labels.PRIMARY_RACE.apply(lambda r: reversed_race_dict[r])
        self.df_labels.dropna(subset=['race'], inplace=True)
        race_encoding = pd.get_dummies(self.df_labels.race)
        self.df_labels[race_encoding.columns] = race_encoding
        self.df_labels = self.df_labels.merge(self.original_labels, how='inner', on='PATIENT')
        self.ann_cols = list(race_encoding.columns)