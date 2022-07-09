from dataclasses import dataclass
import numpy as np
import os
import torch
from torch import nn
import random
import datetime
from sklearn.metrics import roc_auc_score


@dataclass
class Configs:
    # configuration for the entire project
    SEED = 123
    NUM_CLASSES = 14
    VERBOSE = 2
    OUT_FILE = r"log.txt"
    ALL_ANNOTATIONS_COLUMNS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    CHALLENGE_ANNOTATIONS_COLUMNS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    UONES_COLUMNS = ["Edema", "Pleural Effusion", "Atelectasis"]
    UZEROS_COLUMNS = ["Cardiomegaly", "Consolidation"]


def set_seed():
    torch.manual_seed(Configs.SEED)
    random.seed(Configs.SEED)
    np.random.seed(Configs.SEED)


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def vprint(s:str, **print_kargs):
    """
    verbose based vprint
    Verbose=0: suppress vprints
    Verbose=1: vprints to stdout
    Verbose=2: vprint to stdout and to OUT_FILE
    """
    curr_time = str(datetime.datetime.now())[:-10]
    s = f"{curr_time}: {s}"
    if Configs.VERBOSE == 0:
        return
    if Configs.VERBOSE >= 1:
        print(s, **print_kargs)
    if Configs.VERBOSE == 2:
        with open(Configs.OUT_FILE, "a") as file:
            file.write(str(s) + '\n')


def get_time_str():
    time_str = str(datetime.datetime.now())[:-10]
    trans = str.maketrans("-: ","__-")
    return time_str.translate(trans)


def create_checkpoint(model, epoch, i, valid_dataloader, criterion, results, TrainingConfigs):
    try:
        valid_loss, valid_auc = calc_scores(['loss', 'auc'], model, valid_dataloader, criterion)
        results['valid_loss'].append(valid_loss.item())
        results['valid_auc'].append(valid_auc)
    except Exception as e:
        vprint(e)
    # metadata for file naming
    metadata = {
        "epoch": epoch,
        "iter": i,
        "batch_size": TrainingConfigs.BATCH_SIZE,
        "trainLastLoss": np.mean(results["train_loss"][-100:]),
        "validAUC": results["valid_auc"][-1]
    }
    time_str = get_time_str()
    metadata_suffix = '__'.join([f"{k}-{round(v,4)}" for k, v in metadata.items()])
    filename = f"{time_str}__{TrainingConfigs.MODEL_VERSION}__{metadata_suffix}.dict"
    filepath = os.path.join(TrainingConfigs.CHECKPOINT_DIR, filename)
    statedata = {**metadata, **{"model": model.state_dict(), "results": results}}
    torch.save(statedata, filepath)
    vprint(f"{time_str}: Checkpoint Created.")
    vprint('Epoch [%d/%d],   Iter [%d/%d],   Train Loss: %.4f,   Valid Loss: %.4f,   Valid AUC: %.4f'
          % (epoch + 1, TrainingConfigs.EPOCHS,
             i, TrainingConfigs.TRAIN_LOADER_SIZE - 1,
             np.mean(results["train_loss"][-100:]),
             results["valid_loss"][-1],
             results["valid_auc"][-1]),
          end="\n\n")


def calc_scores(scores, model, dataloader, criterion=None):
    labels, outputs = get_metric_tensors(model, dataloader)
    scores_val = []
    if 'loss' in scores:
        scores_val.append(criterion(labels, outputs))
    if 'auc' in scores:
        scores_val.append(auc_score(labels, outputs))
    return scores_val


def get_metric_tensors(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = to_gpu(images)
            outputs = model(images)
            all_outputs.append(outputs)
            all_labels.append(labels)
    all_labels, all_outputs = torch.cat(all_labels).cpu(), torch.cat(all_outputs).cpu()
    return all_labels, all_outputs


def auc_score(labels, outputs, **kargs):
    outputs = torch.sigmoid(outputs)
    return roc_auc_score(labels, outputs, **kargs)


def get_previos_training_place(model, TrainingConfigs):
    if not os.path.isdir(TrainingConfigs.CHECKPOINT_DIR):
        os.mkdir(TrainingConfigs.CHECKPOINT_DIR)
    if TrainingConfigs.TRAINED_MODEL_PATH:
        return load_statedict(model, TrainingConfigs.TRAINED_MODEL_PATH)
    _, _, files = next(os.walk(TrainingConfigs.CHECKPOINT_DIR))
    files = [filename for filename in files if filename.split("__")[1] == TrainingConfigs.MODEL_VERSION]
    if not files:
        results = {
            "train_loss": [-1],
            "valid_loss": [-1],
            "valid_auc": [-1]
        }
        return model, results, 0, -1
    model_filename = files[-1]
    return load_statedict(model, os.path.join(TrainingConfigs.CHECKPOINT_DIR, model_filename))


def load_statedict(model, path):
    statedata = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(statedata['model'])
    statedata['model'] = model
    vprint(f"Loaded model - epoch:{statedata['epoch']}, iter:{statedata['iter']}")
    return [statedata[k] for k in ['model', 'results', 'epoch', 'iter']]

