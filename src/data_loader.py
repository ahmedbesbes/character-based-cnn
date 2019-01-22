import json
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from . import utils

class MyDataset(Dataset):
    def __init__(self, args):

        self.data_path = args.data_path

        self.max_rows = args.max_rows
        self.chunksize = args.chunksize
        self.encoding = args.encoding

        self.vocabulary = list(args.alphabet) + list(args.extra_characters)
        self.number_of_characters = args.number_of_characters + \
            len(args.extra_characters)
        self.max_length = args.max_length
        self.num_classes = args.number_of_classes

        self.preprocessing_steps = args.steps

        self.identity_mat = np.identity(self.number_of_characters)
        texts, labels = [], []

        # chunk your dataframes in small portions
        chunks = pd.read_csv(self.data_path,
                             usecols=[args.text_column, args.label_column],
                             chunksize=self.chunksize,
                             encoding=self.encoding,
                             nrows=self.max_rows)
        for df_chunk in tqdm(chunks):
            df_chunk['processed_text'] = (df_chunk[args.text_column]
                                          .map(lambda text: utils.process_text(self.preprocessing_steps, text)))
            texts += df_chunk['processed_text'].tolist()
            labels += df_chunk[args.label_column].tolist()

        print('data loaded successfully with {0} rows'.format(len(labels)))

        self.texts = texts
        self.labels = labels
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros(
                (self.max_length, self.number_of_characters), dtype=np.float32)
        label = self.labels[index]
        return data, label
