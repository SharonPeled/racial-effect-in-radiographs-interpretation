import numpy as np
import pandas as pd
import os
import torch
from torch import nn
import random
import datetime
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def vprint(s:str, TrainingConfigs, **print_kargs):
    """
    verbose based vprint
    Verbose=0: suppress vprints
    Verbose=1: vprints to stdout
    Verbose=2: vprint to stdout and to OUT_FILE
    """
    curr_time = str(datetime.datetime.now())[:-10]
    s = f"{curr_time}: {s}"
    if TrainingConfigs.VERBOSE == 0:
        return
    if TrainingConfigs.VERBOSE >= 1:
        print(s, **print_kargs)
    if TrainingConfigs.VERBOSE == 2:
        with open(TrainingConfigs.OUT_FILE, "a") as file:
            file.write(str(s) + '\n')
            file.flush()


def start_training_msg(TrainingConfigs):
    vprint('', TrainingConfigs) # newline
    vprint("-" * 100, TrainingConfigs)
    vprint("-" * 100, TrainingConfigs)
    vprint('', TrainingConfigs) # newline
    vprint("Start Training", TrainingConfigs)


def get_time_str():
    time_str = str(datetime.datetime.now())[:-10]
    trans = str.maketrans("-: ","__-")
    return time_str.translate(trans)


def create_checkpoint(model, optimizer, scheduler, criterion, epoch, i, valid_dataloader, results, TrainingConfigs,
                      score_dict, apply_on_outputs=lambda x:x, by_study=None, challenge_ann_only=None):
    try:
        score_vals_dict = calc_scores(score_dict.keys(), model, valid_dataloader, TrainingConfigs, criterion,
                                      apply_on_outputs, by_study, challenge_ann_only)
        for score_name, score_value in score_vals_dict.items():
            results[score_dict[score_name]].append(score_value)
    except Exception as e:
        vprint(str(e), TrainingConfigs)
    # metadata for file naming
    metadata = {
        "epoch": epoch,
        "iter": i,
        "batch_size": TrainingConfigs.BATCH_SIZE,
        "trainLastLoss": np.mean(results["train_loss"][-1]),
        "validAUC": results["valid_auc"][-1]
    }
    training_objects = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion
    }
    time_str = get_time_str()
    metadata_suffix = '__'.join([f"{k}-{round(v,4)}" for k, v in metadata.items()])
    filename = f"{time_str}__{TrainingConfigs.MODEL_VERSION}__{metadata_suffix}.dict"
    filepath = os.path.join(TrainingConfigs.CHECKPOINT_DIR, filename)
    statedata = {**training_objects, **metadata, **{"model": model.state_dict(), "results": results}}
    torch.save(statedata, filepath)
    vprint(f"{time_str}: Checkpoint Created For {TrainingConfigs.MODEL_VERSION}.", TrainingConfigs)
    vprint('Epoch [%d/%d],   Iter [%d/%d],   Train Loss: %.4f,   Valid Loss: %.4f,   Valid AUC: %.4f'
          % (epoch, TrainingConfigs.EPOCHS,
             i, TrainingConfigs.TRAIN_LOADER_SIZE - 1,
             np.mean(results["train_loss"][-100:]),
             results["valid_loss"][-1],
             results["valid_auc"][-1]),
          TrainingConfigs, end="\n\n")


def calc_scores(scores, model, dataloader, TrainingConfigs, criterion=None, apply_on_outputs=lambda x:x,
                by_study=None, challenge_ann_only=None):
    labels, outputs = get_metric_tensors(model, dataloader, TrainingConfigs, apply_on_outputs,
                                         by_study, challenge_ann_only)
    score_vals_dict = {}
    if 'loss' in scores:
        score_vals_dict['loss'] = criterion(labels, outputs).item()
    if 'auc' in scores:
        score_vals_dict['auc'] = auc_score(labels, outputs)
    return score_vals_dict


def get_metric_tensors(model, dataloader, TrainingConfigs, apply_on_outputs, by_study, challenge_ann_only):
    """
    by_study - if None it's ignored. Else should be an agg function to apply on study view outputs.
    For example: max (as in the original paper), mean, min, etc.
    """
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = to_gpu(images)
            outputs = model(images)
            all_outputs.append(outputs)
            all_labels.append(labels)
    model.train()
    all_labels, all_outputs = torch.cat(all_labels).cpu(), torch.cat(all_outputs).cpu()
    if by_study:
        all_labels_df = pd.DataFrame(all_labels, columns=TrainingConfigs.ANNOTATIONS_COLUMNS)
        all_labels_df[["patient_id", "study", "view"]] = dataloader.dataset.get_attributes(columns=
                                                                                           ["patient_id", "study", "view"])
        study_labels = all_labels_df.groupby(["patient_id", "study"]).head(1)[TrainingConfigs.ANNOTATIONS_COLUMNS]
        all_outputs_df = pd.DataFrame(all_outputs, columns=TrainingConfigs.ANNOTATIONS_COLUMNS)
        all_outputs_df[["patient_id", "study", "view"]] = dataloader.dataset.get_attributes(columns=
                                                                                           ["patient_id", "study", "view"])
        study_outputs = all_outputs_df.groupby(["patient_id", "study"]).agg(by_study)[TrainingConfigs.ANNOTATIONS_COLUMNS]
        all_labels, all_outputs = torch.Tensor(study_labels.values), torch.Tensor(study_outputs.values)
    if challenge_ann_only:
        # for disease prediction task only
        col_inds = [i for i, col in enumerate(TrainingConfigs.ANNOTATIONS_COLUMNS)
                       if col in TrainingConfigs.CHALLENGE_ANNOTATIONS_COLUMNS]
        all_labels = all_labels[:,col_inds]
        all_outputs = all_outputs[:,col_inds]
    return all_labels, apply_on_outputs(all_outputs)


def auc_score(labels, outputs, per_class=False, **kargs):
    AUROCs = []
    num_classes = labels.shape[1]
    for i in range(num_classes):
        try:
            # there are classes with single class value in validation
            # which throws an error
            AUROCs.append(roc_auc_score(labels[:, i], outputs[:, i], **kargs))
        except Exception as e:
            print(e)
            AUROCs.append(np.nan)
    if per_class:
        return AUROCs
    return np.mean(AUROCs)

def add_mean_to_list(a):
    a.append(np.nanmean(a))
    return a
    

def get_previous_training_place(model, optimizer, scheduler, criterion, TrainingConfigs):
    if not os.path.isdir(TrainingConfigs.CHECKPOINT_DIR):
        os.mkdir(TrainingConfigs.CHECKPOINT_DIR)
    if TrainingConfigs.TRAINED_MODEL_PATH:
        return load_statedict(model, TrainingConfigs.TRAINED_MODEL_PATH, TrainingConfigs)
    _, _, files = next(os.walk(TrainingConfigs.CHECKPOINT_DIR))
    files = [filename for filename in files if filename.split("__")[1] == TrainingConfigs.MODEL_VERSION]
    if not files:
        results = {
            "train_loss": [-1],
            "valid_loss": [-1],
            "valid_auc": [-1]
        }
        return model, optimizer, scheduler, criterion, results, 0, -1
    files.sort(key=lambda filename: (int(filename.split("epoch-")[1].split("__")[0]),
                                     int(filename.split("iter-")[1].split("__")[0])))
    model_filename = files[-1]
    # ['model', 'optimizer', 'scheduler', 'criterion', 'results', 'epoch', 'iter']
    res = load_statedict(model, os.path.join(TrainingConfigs.CHECKPOINT_DIR, model_filename), TrainingConfigs)
    if res[6] == TrainingConfigs.TRAIN_LOADER_SIZE:
        # if epoch if finished, proceed to the next one
        # avoiding from fast-forward the entire dataset
        res[5] += 1
        res[6] = -1
    # avoiding a bug in pytorch loading/saving optimizer
    if '_last_lr' in res[2].state_dict().keys():
        set_lr_to_optimizer(optimizer, res[2]._last_lr[0])
    else:
        set_lr_to_optimizer(optimizer, TrainingConfigs.LEARNING_RATE)
    res[1] = optimizer
    return res


def set_lr_to_optimizer(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


def load_statedict(model, path, TrainingConfigs, device='gpu'):
    vprint(f"Loading model - {path}", TrainingConfigs)
    if torch.cuda.is_available() and device == 'gpu':
        statedata = torch.load(path, map_location=torch.device('cuda'))
    else:
        statedata = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(statedata['model'])
    statedata['model'] = model
    if model.training:
        return [statedata[k] for k in ['model', 'optimizer', 'scheduler', 'criterion', 'results', 'epoch', 'iter']]
    else:
        return [statedata[k] for k in ['model', 'results', 'epoch', 'iter']]


def load_model(model, model_fullpath):
    path = os.path.join(model_fullpath)
    if not torch.cuda.is_available():
        statedata = torch.load(path, map_location=torch.device('cpu'))
    else:
        statedata = torch.load(path, map_location=torch.device('cuda'))
    model.load_state_dict(statedata['model'])
    return model


def load_portion_of_pretrained_model(full_model, shallow_model):
    pretrained_model_state_dict = full_model.state_dict()
    shallow_model_state_dict = shallow_model.state_dict()
    new_shallow_state_dict = {k: v for k, v in pretrained_model_state_dict.items() if k in shallow_model_state_dict}
    shallow_model_state_dict.update(new_shallow_state_dict)
    shallow_model.load_state_dict(shallow_model_state_dict)
    return shallow_model


def requires_grad_update_by_layer(model, TrainingConfigs, requires_grad):
    """
    Finds <target_layer_name> location in <model> and update requires_grad attribute
    to all <target_layer_name> descendants layers (excluding <target_layer_name> itself).
    Enable to train only part of the network, while the first part remains constant.
    If <target_layer_name> repeats itself in the network, the alg stops on the first encounter (the shallowest
    <target_layer_name> in the nn).
    """
    found = requires_grad_update_by_layer_aux(model, TrainingConfigs.MODEL_VERSION, TrainingConfigs.FREEZING_POINT,
                                              requires_grad, TrainingConfigs)
    if not found:
        raise Exception(f"{TrainingConfigs.FREEZING_POINT} wasn't found in model.")


def requires_grad_update_by_layer_aux(root_module, root_module_name, stop_target_module,
                                      requires_grad, TrainingConfigs):
    if not root_module._modules:
        return False
    if root_module_name == stop_target_module:
        return True
    modules_to_update = []
    modules_to_update_names = []
    for module_name, module in root_module._modules.items():
        found = requires_grad_update_by_layer_aux(module, module_name, stop_target_module, requires_grad,
                                                  TrainingConfigs)
        if found:
            break
        modules_to_update.append(module)
        modules_to_update_names.append(module_name)
    if found:
        vprint(str(modules_to_update_names), TrainingConfigs)
        for module in modules_to_update:
            for p in module.parameters():
                p.requires_grad = requires_grad
        return True
    return None