from dataclasses import dataclass
import numpy as np
import os
import torch
from torch import nn
from torchmetrics.functional import auc
import random
import datetime


@dataclass
class Configs:
    SEED = 123
    NUM_CLASSES = 5
    ANNOTATIONS_COLUMNS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    UONES_COLUMNS = ["Edema", "Pleural Effusion", "Atelectasis"]
    UZEROS_COLUMNS = ["Cardiomegaly", "Consolidation"]


def set_seed():
    torch.manual_seed(Configs.SEED)
    random.seed(Configs.SEED)
    np.random.seed(Configs.SEED)


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def get_time_str():
    time_str = str(datetime.datetime.now())[:-10]
    trans = str.maketrans("-: ","__-")
    return time_str.translate(trans)


def create_checkpoint(model, epoch, i, valid_dataloader, criterion, results, TrainingConfigs):
    valid_loss, valid_auc = calc_auc_score(model, valid_dataloader, criterion)
    results['valid_loss'].append(valid_loss.item())
    results['valid_auc'].append(valid_auc)
    metadata = {
        "epoch": epoch,
        "iter": i,
        "trainLastLoss": np.mean(results["train_loss"][-100:]),
        "validAUC": results["valid_auc"][-1]
    }
    time_str = get_time_str()
    metadata_suffix = '__'.join([f"{k}-{round(v,4)}" for k, v in metadata.items()])
    filename = f"{time_str}__{TrainingConfigs.MODEL_VERSION}__{metadata_suffix}.dict"
    filepath = os.path.join(TrainingConfigs.CHECKPOINT_DIR, filename)
    statedata = {**metadata, **{"model": model.state_dict(), "results": results}}
    torch.save(statedata, filepath)
    print(f"{time_str}: Checkpoint Created.")


def avg_auc(outputs, labels):
    softmax = nn.Softmax(dim=1)
    probas = softmax(outputs).T
    return np.mean([auc(y_proba, y_true, reorder=True) for y_proba, y_true in zip(probas, labels.T)])


def calc_auc_score(model, dataloader, criterion=None):
    all_labels = []
    all_outputs = []
    model.eval()
    for i, (images, labels) in enumerate(dataloader):
        images = to_gpu(images)
        outputs = model(images).cpu()
        all_outputs.append(outputs)
        labels = labels.cpu()
        all_labels.append(labels)
    all_outputs, all_labels = torch.cat(all_outputs), torch.cat(all_labels)
    auc_value = avg_auc(all_outputs, all_labels)
    if auc_value > 1:
        print(all_outputs, all_labels)
        input()
    loss_value = None
    if criterion:
        loss_value = criterion(all_outputs, all_labels)
    model.train()
    return loss_value, auc_value


def get_previos_training_place(model, TrainingConfigs):
    if TrainingConfigs.TRAINED_MODEL_PATH:
        return load_statedict(model, TrainingConfigs.TRAINED_MODEL_PATH)
    _, _, files = next(os.walk(TrainingConfigs.CHECKPOINT_DIR))
    if not files:
        results = {
            "train_loss": [],
            "valid_loss": [],
            "valid_auc": []
        }
        return model, results, 0, -1
    model_filename = [filename for filename in files if filename.split("__")[1] == TrainingConfigs.MODEL_VERSION][-1]
    return load_statedict(model, os.path.join(TrainingConfigs.CHECKPOINT_DIR, model_filename))


def load_statedict(model, path):
    statedata = torch.load(path)
    model.load_state_dict(statedata['model'])
    statedata['model'] = model
    return [statedata[k] for k in ['model', 'results', 'epoch', 'iter']]



