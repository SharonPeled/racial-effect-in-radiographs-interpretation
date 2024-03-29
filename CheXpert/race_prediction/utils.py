from dataclasses import dataclass
from CheXpert.disease_prediction.utils import Configs as DiseaseConfigs
from enum import Enum
from shared_utils import SharedConfigs


@dataclass
class Configs(SharedConfigs):
    RACE_DICT = {
        "White": ["White", "White, non-Hispanic", "White or Caucasian"],
        "Asian": ["Asian", "Asian, non-Hispanic", ],
        "Black": ["Black or African American", "Black, non-Hispanic"],
        "Hispanic": ["Other, Hispanic", "White, Hispanic"]
    }
    ANNOTATIONS_COLUMNS = ['Asian', 'Black', 'Hispanic', 'White']
    NUM_CLASSES = len(ANNOTATIONS_COLUMNS)
    NUM_DISEASE_CLASSES = DiseaseConfigs.NUM_CLASSES


class RaceTrainingMode(Enum):
    Full = 1 # regular training with pretrained model
    PartlyFreezed = 2 # training with first part of the network freezed
    Shallow = 3 # training a model with fewer layers


