from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

USER_ID_FIELD = "userID"
ITEM_ID_FIELD = "movieID"
YEAR_FIELD = "date_year"
TIMESTAMP_FIELD = "timestamp"
RATING_FIELD = "rating"
LABEL_FIELD = "label"


class UserItemPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        User-Item Pair Dataset for training and evaluation.
        Args:
            df (pd.DataFrame): with columns ['userID', 'movieID', 'label']
        """
        self.user_ids = torch.tensor(df[USER_ID_FIELD].values, dtype=torch.long)
        self.item_ids = torch.tensor(df[ITEM_ID_FIELD].values, dtype=torch.long)
        self.labels = torch.tensor(df[LABEL_FIELD].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
