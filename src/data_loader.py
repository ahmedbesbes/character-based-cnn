import json
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import utils


class MyDataset(Dataset):
    def __init__(self, config_path='../config.json'):

        with open(config_path) as f:
            self.config = json.load(f)

        self.data_path = self.config['data']['path']
        self.vocabulary = list(self.config['data']['alphabet'])
        self.max_length = self.config['max_length']
        self.num_classes = self.config['num_of_classes']
        self.preprocessing_steps = self.config['data']['preprocessing_steps']
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []

        # chunk your dataframes in small portions
        chunks = pd.read_csv(
            self.data_path, chunksize=self.config['chunksize'], encoding=self.config['encoding'], nrows=self.config['data']['max_rows'])
        for df_chunk in tqdm(chunks):
            df_chunk['processed_text'] = df_chunk[self.config['data']
                                                  ['text_column']].map(utils.process_text)
            texts += df_chunk['processed_text'].tolist()
            labels += df_chunk[self.config['label_column']].tolist()

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
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(
                self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label
