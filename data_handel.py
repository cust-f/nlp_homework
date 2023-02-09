import csv
import torch
import pandas as pd
import re
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len)
        self.with_label = True if labels is not None else False
        if self.with_label:
            self.labels = list(labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        if self.with_label:
            item['labels'] = torch.tensor(int(self.labels[index]))
        return item

    def __len__(self):
        return len(self.encodings.encodings)

def split_and_load_dataset(data, tokenizer, max_len, batch_size, with_label=True):
    if with_label == False:
        test_dataset = TextClassificationDataset(data["Text"], None, tokenizer, max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return test_dataset, test_loader

    X_train, X_val, y_train, y_val = train_test_split(data["Text"], data["Labels"], random_state=1)
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = TextClassificationDataset(X_val, y_val, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_dataset, val_dataset, train_loader, val_loader

def process_train(file_path):
    dataset = list()
    csv_reader = csv.reader(open(file_path, encoding="utf-8"))
    for row in csv_reader:
        if row[2]:  # 标签存在
            strs = row[1].split('__eou__')
            # 消除多余符号以及特殊字符
            for index in range(len(strs)):
                strs[index] = regex_filter(strs[index])
            labels = list(str(row[2]))
            for data in zip(strs, labels):
                dataset.append(data)
    data = pd.DataFrame(dataset, columns=['Text', 'Labels'])[1:]
    data['Labels'] = data['Labels'].astype(int)
    n_labels = len(set(data.Labels))
    # print(Counter(list(data.Labels)))
    label2idx = dict()
    idx2label = dict()
    for idx, label in enumerate(set(data.Labels)):
        label2idx[label] = idx
        idx2label[idx] = label
    for x in range(n_labels):
        assert label2idx[idx2label[x]] == x

    data['Labels'] = data.Labels.map(lambda x: label2idx[x])

    data = shuffle(data)

    return data, n_labels

def process_test(file_path, with_label=True):
    dataset = list()
    csv_reader = csv.reader(open(file_path, encoding="utf-8"))
    for row in csv_reader:
        if row[2]: # 标签存在
            strs = row[1].split('__eou__')
            for index in range(len(strs)):
                strs[index] = regex_filter(strs[index])
            if with_label:
                labels = list(str(row[2]))
                dataset.append([strs[-1], labels[-1]])
            else:
                dataset.append(strs[-1])


    data = pd.DataFrame(dataset, columns=['Text'])[1:]
    return data


def make_batch(batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device) if 'labels' in batch else None
    return input_ids, attention_mask, labels

# 数据过滤
def regex_filter(s_line):
    # # 剔除英文、数字，以及空格
    special_regex = re.compile(r"[a-zA-Z0-9\s]+")
    # # 剔除英文标点符号和特殊符号
    # en_regex = re.compile(r"[.…{|}#$%&\'()*+,!-_./:~^;<=>?@★●，。]+")
    # # 剔除中文标点符号
    # zn_regex = re.compile(r"[《》！、，“”；：（）【】]+")
    # 剔除英文、数字，以及空格
    # special_regex = re.compile(r"[\s]+")
    # 剔除特殊符号
    en_regex = re.compile(r"[★●]+")

    s_line = special_regex.sub(r"", s_line)
    s_line = en_regex.sub(r"", s_line)

    return s_line
