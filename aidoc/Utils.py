import numpy as np
from torch.utils.data import Dataset


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels):

        if len(seqs)!= len(labels):
            raise ValueError("seqs and labels have different lengths")

        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, item):
        return self.seqs[item], self.labels[item]
