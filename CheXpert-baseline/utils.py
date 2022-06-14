from dataclasses import dataclass
import os

@dataclass
class Configs:
    SEED = 123
    NUM_CLASSES = 5
    CHECKPOINT_DIR = r"\model_checkpoints"
    ANNOTATIONS_COLUMNS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    UONES_COLUMNS = ["Edema", "Pleural Effusion", "Atelectasis"]
    UZEROS_COLUMNS = ["Cardiomegaly", "Consolidation"]////