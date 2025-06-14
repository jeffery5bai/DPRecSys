import os
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
POS_ITEM_FIELD = "pos_item_id"
NEG_ITEM_FIELD = "neg_item_id"

FEATURE_FIELD = ["actorID", "country", "directorID", "genre"]

ENCODED_ACTOR_FIELD = "actorID_idx"
ENCODED_COUNTRY_FIELD = "country_idx"
ENCODED_DIRECTOR_FIELD = "directorID_idx"
ENCODED_GENRE_FIELD = "genre_idx"

ACTOR_DPS_FIELD = "actorID_dps"
COUNTRY_DPS_FIELD = "country_dps"
DIRECTOR_DPS_FIELD = "directorID_dps"
GENRE_DPS_FIELD = "genre_dps"


class TripletDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        User-Item Triplet Dataset for training and validation (BPR).
        Args:
            df (pd.DataFrame): with columns ['userID', 'movieID', 'label'] (optional: encoded features)
        """
        self.features = ["actor", "country", "director", "genre"]
        self.user_ids = torch.tensor(df[USER_ID_FIELD].values, dtype=torch.long)
        self.pos_item_ids = torch.tensor(df[POS_ITEM_FIELD].values, dtype=torch.long)
        self.neg_item_ids = torch.tensor(df[NEG_ITEM_FIELD].values, dtype=torch.long)

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

        # NOTE: item features multihot encoding
        for feat, feature_field in zip(self.features, FEATURE_FIELD):
            vec_attr_name, vec_col_name = f"{feat}_vec", f"{feature_field}_vec"
            wvec_attr_name, wvec_col_name = f"{feat}_wvec", f"{feature_field}_wvec"
            setattr(
                self,
                vec_attr_name,
                (torch.from_numpy(np.stack(df[vec_col_name].values)) if vec_col_name in df.columns else None),
            )
            setattr(
                self,
                wvec_attr_name,
                (
                    torch.from_numpy(np.stack(df[wvec_col_name].values))
                    if wvec_col_name in df.columns
                    else None
                ),
            )

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx) -> Tuple[Optional[torch.tensor]]:
        batch_data = {
            "user": self.user_ids[idx],
            "pos_item": self.pos_item_ids[idx],
            "neg_item": self.neg_item_ids[idx],
            "actor": self.actor_ids[idx],
            "country": self.country_ids[idx],
            "director": self.director_ids[idx],
            "genre": self.genre_ids[idx],
        }

        dps_attrs = {
            "actor_dps": self.actor_dps,
            "country_dps": self.country_dps,
            "director_dps": self.director_dps,
            "genre_dps": self.genre_dps,
        }
        batch_data.update({name: dps[idx] for name, dps in dps_attrs.items() if dps is not None})

        item_vec_attrs = {
            f"{feature}_vec": getattr(self, f"{feature}_vec", None) for feature in self.features
        }
        batch_data.update({name: vec[idx] for name, vec in item_vec_attrs.items() if vec is not None})

        item_wvec_attrs = {
            f"{feature}_wvec": getattr(self, f"{feature}_wvec", None) for feature in self.features
        }
        batch_data.update({name: wvec[idx] for name, wvec in item_wvec_attrs.items() if wvec is not None})

        return batch_data


class UserItemPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        User-Item Pair Dataset for training, validation (BCE) and evaluation (BPR).
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
        batch_data = {
            "user": self.user_ids[idx],
            "item": self.item_ids[idx],
            "label": self.labels[idx],
            "actor": self.actor_ids[idx],
            "country": self.country_ids[idx],
            "director": self.director_ids[idx],
            "genre": self.genre_ids[idx],
        }

        dps_attrs = {
            "actor_dps": self.actor_dps,
            "country_dps": self.country_dps,
            "director_dps": self.director_dps,
            "genre_dps": self.genre_dps,
        }

        batch_data.update({name: dps[idx] for name, dps in dps_attrs.items() if dps is not None})

        return batch_data


class TripletDatasetFromCached(Dataset):
    def __init__(self, split: str = "train", cache_dir: str = "../experiments/cache"):
        self.split = split
        self.cache_dir = cache_dir

        # Load triplet (user, pos_item, neg_item)
        triplet_path = os.path.join(cache_dir, f"{split}_triplet_with_dps.parquet")
        self.triplet_df = pd.read_parquet(triplet_path)

        # Load IDs
        self.actor_ids = np.load(os.path.join(cache_dir, f"{split}_actor_ids.npy"), allow_pickle=True)
        self.director_ids = np.load(os.path.join(cache_dir, f"{split}_director_ids.npy"), allow_pickle=True)
        self.country_ids = np.load(os.path.join(cache_dir, f"{split}_country_ids.npy"), allow_pickle=True)
        self.genre_ids = np.load(os.path.join(cache_dir, f"{split}_genre_ids.npy"), allow_pickle=True)

        # Load DPS
        self.actor_dps = np.load(os.path.join(cache_dir, f"{split}_actor_dps.npy"))
        self.director_dps = np.load(os.path.join(cache_dir, f"{split}_director_dps.npy"))
        self.country_dps = np.load(os.path.join(cache_dir, f"{split}_country_dps.npy"))
        self.genre_dps = np.load(os.path.join(cache_dir, f"{split}_genre_dps.npy"))

        # Load VEC / WVEC
        self.actor_vec = np.load(os.path.join(cache_dir, f"{split}_actor_vec.npy"))
        self.director_vec = np.load(os.path.join(cache_dir, f"{split}_director_vec.npy"))
        self.country_vec = np.load(os.path.join(cache_dir, f"{split}_country_vec.npy"))
        self.genre_vec = np.load(os.path.join(cache_dir, f"{split}_genre_vec.npy"))

        self.actor_wvec = np.load(os.path.join(cache_dir, f"{split}_actor_wvec.npy"))
        self.director_wvec = np.load(os.path.join(cache_dir, f"{split}_director_wvec.npy"))
        self.country_wvec = np.load(os.path.join(cache_dir, f"{split}_country_wvec.npy"))
        self.genre_wvec = np.load(os.path.join(cache_dir, f"{split}_genre_wvec.npy"))

        assert len(self.triplet_df) == len(self.actor_ids), "Mismatch in sample size!"

    def __len__(self):
        return len(self.triplet_df)

    def __getitem__(self, idx):
        row = self.triplet_df.iloc[idx]

        batch = {
            "userID": int(row["userID"]),
            "pos_item_id": int(row["pos_item_id"]),
            "neg_item_id": int(row["neg_item_id"]),
            # IDs
            "actor_ids": torch.from_numpy(self.actor_ids[idx]).long(),
            "director_ids": torch.from_numpy(self.director_ids[idx]).long(),
            "country_ids": torch.from_numpy(self.country_ids[idx]).long(),
            "genre_ids": torch.from_numpy(self.genre_ids[idx]).long(),
            # DPS
            "actor_dps": torch.from_numpy(self.actor_dps[idx]).float(),
            "director_dps": torch.from_numpy(self.director_dps[idx]).float(),
            "country_dps": torch.from_numpy(self.country_dps[idx]).float(),
            "genre_dps": torch.from_numpy(self.genre_dps[idx]).float(),
            # Multihot vec
            "actor_vec": torch.from_numpy(self.actor_vec[idx]).float(),
            "director_vec": torch.from_numpy(self.director_vec[idx]).float(),
            "country_vec": torch.from_numpy(self.country_vec[idx]).float(),
            "genre_vec": torch.from_numpy(self.genre_vec[idx]).float(),
            # Weighted vec
            "actor_wvec": torch.from_numpy(self.actor_wvec[idx]).float(),
            "director_wvec": torch.from_numpy(self.director_wvec[idx]).float(),
            "country_wvec": torch.from_numpy(self.country_wvec[idx]).float(),
            "genre_wvec": torch.from_numpy(self.genre_wvec[idx]).float(),
        }

        return batch
