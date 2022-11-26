import numpy as np
import torch
import pandas as pd

from os.path import join
from sklearn.metrics import hamming_loss, accuracy_score
from tqdm import tqdm
from datasets import Dataset
from torch import tensor


def hamming_score(y_true, y_pred):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])  # keep indexes of 1
        set_predictions = set(np.where(y_pred[i])[0])  # keep indexes of 1
        if len(set_true) == 0 and len(set_predictions) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_predictions)) / float(len(set_true.union(set_predictions)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


# Explained
y_true = np.array([[0, 1], [0, 1], [1, 1], [1, 1]])
y_pred = np.array([[0, 1], [0, 0], [1, 1], [0, 1]])
hamming_loss(y_true, y_pred)  # fraction of label incorrectly predicted : 2/8
hamming_score(y_true, y_pred)  # mean on labels of ones correctly predicted : 2.5/4
accuracy_score(y_true, y_pred)  # proportion of exact match : 2/4


def add_prefix_to_x(df, prefix, x_name, initial_x_name):
    new_x = []
    for i in range(df.shape[0]):
        new_x.append(prefix + df.loc[i, initial_x_name])
    df[x_name] = new_x
    return df


def extract_value(df, transition_word=" AND ", y_name="y_dictionnary"):
    y_value = []
    for v in tqdm(df[y_name]):
        y = ""
        for action in v.split(";"):
            if y != "":
                y += transition_word + eval(action).get(list(eval(action).keys())[0])
            elif action == "autres":
                y += "autres"
            else:
                y += eval(action).get(list(eval(action).keys())[0])
        y_value.append(y)
    return y_value


def map_labels(labels, output_best_model):
    map_ = {"num_class": [], "label": []}
    for i in range(labels.num_classes):
        map_["num_class"].append("LABEL_" + str(i))
        map_["label"].append(labels.int2str(i))
    map_ = pd.DataFrame(map_)
    map_.to_csv(join(output_best_model, "map_labels_config.csv"))


def adjust_probability_predictions(predictions, probability):
    filter_predictions = []
    for p in predictions:
        tmp_p = ""
        for value, score in zip(p[0], p[1]):
            if tmp_p == "":
                tmp_p += value
            elif score > probability:
                tmp_p += " AND " + value
        filter_predictions.append(tmp_p)
    return filter_predictions


def filter_none_data(df, x_name):
    ls_index = []
    for i in df.index:
        if str(df.loc[i][x_name]) != "nan":
            ls_index.append(i)

    return df.loc[df.index.isin(ls_index)]


def find_metric(metric_name):
    if metric_name == "f1":
        metric_name = "f1-macro"
    assert len(metric_name.split("-")) == 2, "you must write f1-youraverage for f1 metric"
    metric, average = metric_name.split("-")
    return metric, average


def dict_to_ls_dict(metrics):
    metrics_ = {}
    for i in zip(metrics, metrics.values()):
        metrics_[i[0]] = [i[1]]
    return metrics_


def preprocess(df, x_name, label_name, model, tokenizer):
    df = filter_none_data(df, x_name)
    x = df[x_name].tolist()
    df_tokenized = tokenizer(x, truncation=True, padding=True, return_tensors="pt")
    df_tokenized["labels"] = [model.config.label2id.get(l) for l in df[label_name].tolist()]
    # df_tokenized["labels"] = [labels.str2int(l) for l in df[label_name].tolist()]
    return Dataset.from_dict(df_tokenized)


def preprocess_t5(inputs, outputs, tokenizer):
    model_inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, padding=True, truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class t5_dataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.examples.items()}

    def __len__(self):
        return len(self.examples["input_ids"])
