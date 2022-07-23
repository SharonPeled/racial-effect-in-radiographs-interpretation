from dataclasses import dataclass
from shared_utils import SharedConfigs


@dataclass
class Configs(SharedConfigs):
    ALL_ANNOTATIONS_COLUMNS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    CHALLENGE_ANNOTATIONS_COLUMNS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    UONES_COLUMNS = ["Edema", "Pleural Effusion", "Atelectasis"]
    UZEROS_COLUMNS = ["Cardiomegaly", "Consolidation"]
    ANNOTATIONS_COLUMNS = CHALLENGE_ANNOTATIONS_COLUMNS
    NUM_CLASSES = len(ANNOTATIONS_COLUMNS)

















