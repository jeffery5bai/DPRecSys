from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

USER_ID_FIELD = "userID"
ITEM_ID_FIELD = "movieID"
YEAR_FIELD = "date_year"
TIMESTAMP_FIELD = "timestamp"
RATING_FIELD = "rating"
LABEL_FIELD = "label"

ENCODED_ACTOR_FIELD = "actorID_idx"
ENCODED_COUNTRY_FIELD = "country_idx"
ENCODED_DIRECTOR_FIELD = "directorID_idx"
ENCODED_GENRE_FIELD = "genre_idx"

ACTOR_DPS_FIELD = "actorID_dps"
COUNTRY_DPS_FIELD = "country_dps"
DIRECTOR_DPS_FIELD = "directorID_dps"
GENRE_DPS_FIELD = "genre_dps"

class UserItemPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        User-Item Pair Dataset for training and evaluation.
        Args:
            df (pd.DataFrame): with columns ['userID', 'movieID', 'label'] (optional: encoded features)
        """
        self.user_ids = torch.tensor(df[USER_ID_FIELD].values, dtype=torch.long)
        self.item_ids = torch.tensor(df[ITEM_ID_FIELD].values, dtype=torch.long)
        self.labels = torch.tensor(df[LABEL_FIELD].values, dtype=torch.float)

        self.actor_ids = (
            torch.tensor(
                np.array(df[ENCODED_ACTOR_FIELD].apply(lambda x: np.array(x)).tolist()), dtype=torch.long
            )
            if ENCODED_ACTOR_FIELD in df.columns
            else None
        )
        self.country_ids = (
            torch.tensor(df[ENCODED_COUNTRY_FIELD].values, dtype=torch.long)
            if ENCODED_COUNTRY_FIELD in df.columns
            else None
        )
        self.director_ids = (
            torch.tensor(df[ENCODED_DIRECTOR_FIELD].values, dtype=torch.long)
            if ENCODED_DIRECTOR_FIELD in df.columns
            else None
        )
        self.genre_ids = (
            torch.tensor(
                np.array(df[ENCODED_GENRE_FIELD].apply(lambda x: np.array(x)).tolist()), dtype=torch.long
            )
            if ENCODED_GENRE_FIELD in df.columns
            else None
        )

        # NOTE: diversity preference scale (DPS) labels
        self.actor_dps = (
            torch.tensor(df[ACTOR_DPS_FIELD].values, dtype=torch.float)
            if ACTOR_DPS_FIELD in df.columns
            else None
        )
        self.country_dps = (
            torch.tensor(df[COUNTRY_DPS_FIELD].values, dtype=torch.float)
            if COUNTRY_DPS_FIELD in df.columns
            else None
        )
        self.director_dps = (
            torch.tensor(df[DIRECTOR_DPS_FIELD].values, dtype=torch.float)
            if DIRECTOR_DPS_FIELD in df.columns
            else None
        )
        self.genre_dps = (
            torch.tensor(df[GENRE_DPS_FIELD].values, dtype=torch.float)
            if GENRE_DPS_FIELD in df.columns
            else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[Optional[torch.tensor]]:
        return {
            "user": self.user_ids[idx],
            "item": self.item_ids[idx],
            "label": self.labels[idx],
            "actor": self.actor_ids[idx],
            "country": self.country_ids[idx],
            "director": self.director_ids[idx],
            "genre": self.genre_ids[idx],
            "actor_dps": self.actor_dps[idx],
            "country_dps": self.country_dps[idx],
            "director_dps": self.director_dps[idx],
            "genre_dps": self.genre_dps[idx],
        }
