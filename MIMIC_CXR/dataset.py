from generic_dataset import GenericDataset
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from MIMIC_CXR.utils import Configs
from MIMIC_CXR import utils
from shared_utils import vprint, Mode
import datetime
from tqdm.notebook import tqdm
from CheXpert.race_prediction.dataset import CheXpertRaceDataset
from CheXpert.disease_prediction.dataset import CheXpertDiseaseDataset
import shared_utils


class CXRDataset(GenericDataset):
    def __init__(self, mode, data_dir, labels_filename, transform=None, target_transform=None):
        self.mode = mode
        labels_path = os.path.join(data_dir, labels_filename)
        if not os.path.isfile(labels_path):
            raise Exception("Label file not exists.")
        super().__init__(data_dir, labels_filename, transform, target_transform)
        self.df_labels = pd.read_csv(labels_path)
        if mode == Mode.Disease:
            self.df_labels = CheXpertDiseaseDataset.impute_uncertainties(self.df_labels)
            self.ann_cols = Configs.DISEASE_ANNOTATIONS_COLUMNS
        if mode == Mode.Race:
            self.df_labels = CheXpertRaceDataset.generate_race_dummies(self.df_labels, 'ethnicity', Configs.RACE_DICT)
            self.ann_cols = Configs.RACE_ANNOTATIONS_COLUMNS
        self.df_labels.reset_index(drop=True, inplace=True)

    @classmethod
    def download_dataset(cls, group_sample_size, mode, data_dir, labels_filename, cxr_chexpert_labels_filename, admissions_filename,
                         split_filename, patients_filename, transform=None, target_transform=None):
        """
        :param group_sample_size: if int then sample from each group group_sample_size samples. If dict, then
        should be full_group_str -> int, sample from each group different number of samples. The full_group_str should
        be of form race_age_gender.
        """
        df_cxp = pd.read_csv(os.path.join(data_dir, cxr_chexpert_labels_filename))
        df_adm = pd.read_csv(os.path.join(data_dir, admissions_filename))
        df_split = pd.read_csv(os.path.join(data_dir, split_filename))
        df_patients = pd.read_csv(os.path.join(data_dir, patients_filename))
        df_cxr_joined = df_cxp.merge(df_adm, on='subject_id').\
            merge(df_patients, on='subject_id').merge(df_split, on=['subject_id', 'study_id'])
        df_cxr_joined['race'] = df_cxr_joined.ethnicity.replace(Configs.RACE_DICT_REVERSED_FULL)
        df_cxr_joined['age'] = df_cxr_joined.anchor_age.apply(shared_utils.age_to_age_group)
        df_cxr_demo = df_cxr_joined[
            ['subject_id', 'study_id', 'split', 'dicom_id'] + ['ethnicity', 'race', 'age', 'gender'] + Configs.DISEASE_ANNOTATIONS_COLUMNS]
        is_patient_all_confident_labels = df_cxr_demo.groupby(['subject_id'])[Configs.DISEASE_ANNOTATIONS_COLUMNS].apply(lambda s: (s==-1).sum(axis=1).sum(axis=0)==0)
        confidence_patients = np.array(is_patient_all_confident_labels[is_patient_all_confident_labels].index)
        df_cxr_demo = df_cxr_demo[df_cxr_demo.subject_id.isin(confidence_patients)]
        df_temp = (df_cxr_demo.groupby('subject_id')[['race', 'age', 'gender']].nunique() == 1).all(axis=1)
        valid_subject_ids = df_temp[df_temp.values].index
        df_cxr_demo = df_cxr_demo[df_cxr_demo.subject_id.isin(valid_subject_ids)].drop_duplicates()
        df_cxr_demo = CXRDataset.sample(df_cxr_demo, group_sample_size)
        successes, failures = CXRDataset.download(data_dir, df_cxr_demo)
        df_cxr_demo = df_cxr_demo.merge(successes, on=['subject_id', 'study_id', 'dicom_id'])
        df_cxr_demo['img_path'] = df_cxr_demo.apply(lambda row: os.path.join(data_dir,
                                                    f"physionet.org/files/mimic-cxr-jpg/2.0.0/files/p{row['folder_number']}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg")
                                                    , axis=1)
        df_cxr_demo.to_csv(os.path.join(data_dir, labels_filename), index=False)
        return cls(mode, data_dir, labels_filename, transform, target_transform)

    @staticmethod
    def download(data_dir, df_cxr):
        vprint("Enter username", utils.Configs)
        username = input()
        vprint("Enter password", utils.Configs)
        password = input()
        directory = data_dir
        folder_numbers = list(range(10,20))
        successes = []
        failures = []
        for patient_id, study_id, dicom_id in tqdm(df_cxr[['subject_id', 'study_id', 'dicom_id']].values):
            done = False
            for folder_number in folder_numbers:
                cmd = f"""wget -r -N -c -np --user {username} --password {password} -P {directory} https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/p{folder_number}/p{patient_id}/s{study_id}/{dicom_id}.jpg"""
                res = os.system(cmd)
                if res == 0:
                    successes.append((patient_id, study_id, dicom_id, folder_number))
                    done = True
                    break
            if not done:
                failures.append((patient_id, study_id, dicom_id, -1))
        curr_time = str(datetime.datetime.now())[:-10]
        successes = pd.DataFrame(data=successes, columns=['subject_id', 'study_id', 'dicom_id', 'folder_number'])
        failures = pd.DataFrame(data=failures, columns=['subject_id', 'study_id', 'dicom_id', 'folder_number'])
        successes.to_csv(os.path.join(data_dir, f"successes_{curr_time}.csv"), index=False)
        failures.to_csv(os.path.join(data_dir, f"failures{curr_time}.csv"), index=False)
        vprint(f"MIMIC-CXR successes downloads: {len(successes)}", Configs)
        vprint(f"MIMIC-CXR failed downloads: {len(failures)}", Configs)
        return successes, failures

    @staticmethod
    def sample(df_cxr_demo, group_sample_size):
        if isinstance(group_sample_size, int):
            return df_cxr_demo.groupby(['race', 'age', 'gender']).sample(n=group_sample_size,
                                                                                replace=False,
                                                                                random_state=Configs.SEED)
        if not isinstance(group_sample_size, dict):
            raise NotImplementedError
        groups = [tuple(group.split('_')) for group in group_sample_size.keys()]
        df_list = []
        for attrs, df_group in df_cxr_demo.groupby(['race', 'age', 'gender']):
            if attrs in groups:
                key = '_'.join(attrs)
                df_list.append(df_group.sample(group_sample_size[key], random_state=Configs.SEED))
        return pd.concat(df_list, ignore_index=True)

