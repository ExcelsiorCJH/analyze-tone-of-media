import pickle
from collections import Counter, defaultdict
from functools import partial

import torch
from torch.utils.data import Dataset

from utils.types_ import *


class NewsDataset(Dataset):
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        keyword = self.data[idx][0]
        label = self.data[idx][1]
        text = self.data[idx][4]
        return text, label, keyword


def collate_fn(batch, word_index, labels_dict, max_len=200):
    texts = [entry[0] for entry in batch]
    labels = [entry[1] for entry in batch]
    keywords = [entry[2] for entry in batch]

    sequences = []  # [[word_index.get(word, 1) for word in text] for text in texts]
    for text in texts:
        if len(text) > max_len:
            sequence = [word_index.get(word, 1) for word in text[:max_len]]
        else:
            sequence = [0 for _ in range(max_len - len(text))] + [
                word_index.get(word, 1) for word in text
            ]

        sequences.append(sequence)

    labels = [labels_dict[label] for label in labels]

    sequences = torch.LongTensor(sequences)
    labels = torch.LongTensor(labels)
    return sequences, labels, keywords


# class NewsDataset(Dataset):
#     def __init__(
#         self,
#         path: str,
#         word_index: Dict,
#         max_len: int = 256,
#         labels: List = ["조선일보", "동아일보", "경향신문", "한겨레"],
#     ):
#         self.word_index = word_index
#         self.max_len = max_len
#         self.label_dict = {label: idx for idx, label in enumerate(labels)}

#         with open(path, "rb") as f:
#             self.data = pickle.load(f)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         keyword = self.data[idx][0]
#         label = self.data[idx][1]
#         label = self.label_dict[label]
#         text = self.data[idx][4]
#         sequence = self.text_to_sequence(text)
#         sequence = self.pad_sequence(sequence)
#         return sequence, label, keyword

#     def text_to_sequence(self, text):
#         sequence = [self.word_index.get(word, 1) for word in text]
#         return sequence

#     def pad_sequence(self, sequence):
#         if len(sequence) > self.max_len:
#             sequence = sequence[: self.max_len]
#         else:
#             sequence = [0] * (self.max_len - len(sequence)) + sequence
#         return sequence
