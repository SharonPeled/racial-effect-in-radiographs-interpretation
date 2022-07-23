from dataclasses import dataclass
from CheXpert.disease_prediction.utils import Configs as disease_configs
from enum import Enum


@dataclass
class Configs:
    # configuration for the race prediction task
    SEED = 123
    VERBOSE = 2
    OUT_FILE = r"log.txt"
    RACE_DICT = {
        "White": ["White", "White, non-Hispanic", "White or Caucasian"],
        "Asian": ["Asian", "Asian, non-Hispanic", ],
        "Black": ["Black or African American", "Black, non-Hispanic"],
        "Hispanic": ["Other, Hispanic", "White, Hispanic"]
    }
    ANNOTATIONS_COLUMNS = ['Asian', 'Black', 'Hispanic', 'White']
    NUM_CLASSES = len(ANNOTATIONS_COLUMNS)
    NUM_DISEASE_CLASSES = disease_configs.NUM_CLASSES


class RaceTrainingMode(Enum):
    Full = 1 # regular training with pretrained model
    PartlyFreezed = 2 # training with first part of the network freezed
    Shallow = 3


