from generic_dataset import GenericDataset
import pandas as pd
import os
from collections import defaultdict
from utils import Configs, Mode
import utils

class CXRDataset(GenericDataset):
    def __init__(self, mode, data_dir, labels_filename, transform=None, target_transform=None):
        self.mode = mode
        labels_path = os.path.join(self.data_dir, labels_filename)
        if not os.path.isfile(labels_path):
            raise Exception("Label file not exists.")
        super().__init__(data_dir, labels_filename, transform, target_transform)
        self.df_labels = pd.DataFrame(labels_path)
        if mode == Mode.Disease:
            self.ann_cols = Configs.DISEASE_NUM_CLASSES
        if mode == Mode.Race:
            self.ann_cols = Configs.RACE_ANNOTATIONS_COLUMNS

    @classmethod
    def download_dataset(cls, group_sample_size, data_dir, labels_filename, mode, chexpert_labels=None,
                         admissions_filename=None, split_filename=None, patients_filename=None,
                         transform=None, target_transform=None):
        df_cxp = pd.read_csv(os.path.join(data_dir, chexpert_labels))
        df_adm = pd.read_csv(os.path.join(data_dir, admissions_filename))
        df_split = pd.read_csv(os.path.join(data_dir, split_filename))
        df_patients = pd.read_csv(os.path.join(data_dir, patients_filename))
        df_cxr_joined = df_cxp.merge(df_adm, on='subject_id').merge(df_patients, on='subject_id').merge(df_split,
                                                                                                        on=[
                                                                                                            'subject_id',
                                                                                                            'study_id'])
        df_cxr_joined['race'] = df_cxr_joined.ethnicity.replace(Configs.RACE_DICT)
        df_cxr_joined['age'] = df_cxr_joined.anchor_age.apply(utils.age_to_age_group)
        df_cxr_demo = df_cxr_joined[
            ['subject_id', 'study_id', 'split'] + ['race', 'age', 'gender'] + Configs.DISEASE_NUM_CLASSES]
        df_temp = (df_cxr_demo.groupby('subject_id')[['race', 'age', 'gender']].nunique() == 1).all(axis=1)
        valid_subject_ids = df_temp[df_temp.values].index
        df_cxr_demo = df_cxr_demo[df_cxr_demo.subject_id.isin(valid_subject_ids)].drop_duplicates()
        df_cxr_demo = df_cxr_demo.groupby(['race', 'age', 'gender']).sample(n=group_sample_size,
                                                                            replace=False, random_state=Configs.SEED)
        



