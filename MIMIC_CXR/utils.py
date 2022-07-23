from dataclasses import dataclass
from enum import Enum
from CheXpert.disease_prediction.utils import Configs as disease_congifs
from CheXpert.race_prediction.utils import Configs as race_congifs


class Mode(Enum):
    Disease = 1
    Race = 2


@dataclass
class Configs:
    # configuration for the disease prediction task
    SEED = 123
    VERBOSE = 2
    OUT_FILE = r"log.txt"
    RACE_DICT_REVERSED_FULL = {
        "WHITE": "White",
        "OTHER": "Other",
        "ASIAN": "Asian",
        "BLACK/AFRICAN AMERICAN": "Black",
        "HISPANIC/LATINO": "Hispanic",
        "UNKNOWN": "Other",
        "UNABLE TO OBTAIN": "Other",
        "AMERICAN INDIAN/ALASKA NATIVE": "Other"
    }
    RACE_DICT = {
        "White": ["WHITE", ],
        "Asian": ["ASIAN", ],
        "Black": ["BLACK/AFRICAN AMERICAN", ],
        "Hispanic": ["HISPANIC/LATINO", ]
    }
    CXR_FILENAMES = {
        "cxr_chexpert_labels_filename": "mimic-cxr-2.0.0-chexpert.csv",
        "admissions_filename": "admissions.csv",
        "split_filename": "mimic-cxr-2.0.0-split.csv",
        "patients_filename": "patients.csv"
    }
    DISEASE_ANNOTATIONS_COLUMNS = disease_congifs.CHALLENGE_ANNOTATIONS_COLUMNS
    DISEASE_NUM_CLASSES = len(DISEASE_ANNOTATIONS_COLUMNS)
    RACE_ANNOTATIONS_COLUMNS = race_congifs.ANNOTATIONS_COLUMNS
    RACE_NUM_CLASSES = len(RACE_ANNOTATIONS_COLUMNS)


def age_to_age_group(age):
    if age < 40:
        return '20-40'
    elif age < 70:
        return '40-70'
    return '70-90'















