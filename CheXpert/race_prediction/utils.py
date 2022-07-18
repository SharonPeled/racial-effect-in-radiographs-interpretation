from dataclasses import dataclass
from CheXpert.disease_prediction.utils import Configs as disease_configs


@dataclass
class Configs:
    # configuration for the race prediction task
    SEED = 123
    VERBOSE = 2
    OUT_FILE = r"log.txt"
    RACE_DICT = agg_dict = {
        "White": ["White", "White, non-Hispanic", "White or Caucasian"],
        "Asian": ["Asian", "Asian, non-Hispanic", ],
        "Black": ["Black or African American", "Black, non-Hispanic"],
        "Hispanic": ["Other, Hispanic", "White, Hispanic"]
    }
    NUM_CLASSES = len(RACE_DICT.keys())
    NUM_DISEASE_CLASSES = disease_configs.NUM_CLASSES