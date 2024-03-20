import json
import random
import pandas as pd
from pathlib import Path
import torch.nn.functional
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ShroomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.data = self._load_data()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.data)

    def _load_data(self):
        filenames = ["test.model-agnostic.json", "test.model-aware.json", "train.model-agnostic.json", "train.model-aware.json"]
        raw_data = [json.loads((self.root_dir / filename).read_text()) for filename in filenames]
        flatten = lambda arr: [x for sub_arr in arr for x in sub_arr]
        raw_data = flatten(raw_data)

        data = []
        separator = '<separator>'

        for entry in raw_data:
            concatenated = entry['src'] + separator + entry['hyp']
            label = int(entry.get('p(Hallucination)'))
            label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2).tolist()
            data.append({'src' : entry['src'], 'hyp' : entry['hyp'], 'concatenated' : concatenated, 'label' : label})

        return data

    def get_split(self):
        random.shuffle(self.data)
        train_data, val_data = train_test_split(self.data, test_size=0.1)

        df = pd.DataFrame(train_data)
        non_factuals = df['label'].apply(lambda x: x == [0, 1])
        factuals = df['label'].apply(lambda x: x == [1, 0])
        ratio_of_non_factuals = non_factuals.sum() / factuals.sum()
        df = df.drop(df[factuals].sample(frac=(1 - ratio_of_non_factuals)).index)
        train_data = df.to_dict(orient='records')

        df = pd.DataFrame(val_data)
        non_factuals = df['label'].apply(lambda x: x == [0, 1])
        factuals = df['label'].apply(lambda x: x == [1, 0])
        ratio_of_non_factuals = non_factuals.sum() / factuals.sum()
        df = df.drop(df[factuals].sample(frac=(1 - ratio_of_non_factuals)).index)
        val_data = df.to_dict(orient='records')

        return train_data, val_data