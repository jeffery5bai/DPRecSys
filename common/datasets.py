import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

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
    def __init__(self, split: str = "train", cache_dir: str = "cache"):
        self.split = split
        self.cache_dir = cache_dir

        # Load triplet (user, pos_item, neg_item)
        triplet_path = os.path.join(cache_dir, f"{split}_triplet_with_dps.parquet")
        self.triplet_df = pd.read_parquet(triplet_path)

        # Load IDs
        self.actor_ids = np.load(os.path.join(cache_dir, f"{split}_actor_ids.npy"), mmap_mode="r")
        self.director_ids = np.load(os.path.join(cache_dir, f"{split}_director_ids.npy"), mmap_mode="r")
        self.country_ids = np.load(os.path.join(cache_dir, f"{split}_country_ids.npy"), mmap_mode="r")
        self.genre_ids = np.load(os.path.join(cache_dir, f"{split}_genre_ids.npy"), mmap_mode="r")

        # Load DPS
        self.actor_dps = np.load(os.path.join(cache_dir, f"{split}_actor_dps.npy"), mmap_mode="r")
        self.director_dps = np.load(os.path.join(cache_dir, f"{split}_director_dps.npy"), mmap_mode="r")
        self.country_dps = np.load(os.path.join(cache_dir, f"{split}_country_dps.npy"), mmap_mode="r")
        self.genre_dps = np.load(os.path.join(cache_dir, f"{split}_genre_dps.npy"), mmap_mode="r")

        # Load VEC / WVEC
        self.actor_vec = np.load(os.path.join(cache_dir, f"{split}_actor_vec.npy"), mmap_mode="r")
        self.director_vec = np.load(os.path.join(cache_dir, f"{split}_director_vec.npy"), mmap_mode="r")
        self.country_vec = np.load(os.path.join(cache_dir, f"{split}_country_vec.npy"), mmap_mode="r")
        self.genre_vec = np.load(os.path.join(cache_dir, f"{split}_genre_vec.npy"), mmap_mode="r")

        self.actor_wvec = np.load(os.path.join(cache_dir, f"{split}_actor_wvec.npy"), mmap_mode="r")
        self.director_wvec = np.load(os.path.join(cache_dir, f"{split}_director_wvec.npy"), mmap_mode="r")
        self.country_wvec = np.load(os.path.join(cache_dir, f"{split}_country_wvec.npy"), mmap_mode="r")
        self.genre_wvec = np.load(os.path.join(cache_dir, f"{split}_genre_wvec.npy"), mmap_mode="r")

        assert len(self.triplet_df) == len(self.actor_ids), "Mismatch in sample size!"

    def __len__(self):
        return len(self.triplet_df)

    def __getitem__(self, idx):
        row = self.triplet_df.iloc[idx]

        batch = {
            "user": int(row["userID"]),
            "pos_item": int(row["pos_item_id"]),
            "neg_item": int(row["neg_item_id"]),
            # IDs
            "actor": torch.from_numpy(self.actor_ids[idx]).long(),
            "country": torch.tensor(self.country_ids[idx]).long(),
            "director": torch.tensor(self.director_ids[idx]).long(),  # already 1D
            "genre": torch.from_numpy(self.genre_ids[idx]).long(),
            # DPS
            "actor_dps": torch.tensor(self.actor_dps[idx]).float(),
            "director_dps": torch.tensor(self.director_dps[idx]).float(),
            "country_dps": torch.tensor(self.country_dps[idx]).float(),
            "genre_dps": torch.tensor(self.genre_dps[idx]).float(),
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


class UserItemPairDatasetFromCached(Dataset):
    def __init__(self, split: str = "prediction", cache_dir: str = "cache"):
        self.split = split
        self.cache_dir = cache_dir

        # Load pair (user, item, label)
        pair_path = os.path.join(cache_dir, f"{split}_pool_df.parquet")
        self.pair_df = pd.read_parquet(pair_path)

        # Load IDs
        self.actor_ids = np.load(os.path.join(cache_dir, f"{split}_actor_ids.npy"), mmap_mode="r")
        self.director_ids = np.load(os.path.join(cache_dir, f"{split}_director_ids.npy"), mmap_mode="r")
        self.country_ids = np.load(os.path.join(cache_dir, f"{split}_country_ids.npy"), mmap_mode="r")
        self.genre_ids = np.load(os.path.join(cache_dir, f"{split}_genre_ids.npy"), mmap_mode="r")

        assert len(self.pair_df) == len(self.actor_ids), "Mismatch in sample size!"

    def __len__(self):
        return len(self.pair_df)

    def __getitem__(self, idx):
        row = self.pair_df.iloc[idx]

        batch = {
            "user": int(row["userID"]),
            "item": int(row["movieID"]),
            "label": float(row["label"]),
            # IDs
            "actor": torch.from_numpy(self.actor_ids[idx]).long(),
            "country": torch.tensor(self.country_ids[idx]).long(),
            "director": torch.tensor(self.director_ids[idx]).long(),  # already 1D
            "genre": torch.from_numpy(self.genre_ids[idx]).long(),
        }
        return batch


class UserPosItemSampler(Sampler):
    def __init__(
        self,
        user_to_pos_items,
        user_pos_to_indices,
        batch_size=1024,
        min_pos_items=2,
        max_pos_items=20,
        max_triplets_per_user=None,
        buffer=0,
        drop_last=True,
        seed=42,
    ):
        self.seed = seed
        self.drop_last = drop_last
        self.buffer = buffer
        self.user_to_pos_items = {u: list(p) for u, p in user_to_pos_items.items()}
        self.user_pos_to_indices = {k: list(v) for k, v in user_pos_to_indices.items()}
        self.batch_size = batch_size
        self.min_pos_items = min_pos_items
        self.max_pos_items = max_pos_items
        self.max_triplets_per_user = max_triplets_per_user or batch_size // 10  # Default to 10% of batch size

        self.users = list(self.user_to_pos_items.keys())
        self.total_instances = sum(len(v) for v in user_pos_to_indices.values())

    def __iter__(self):
        rnd = random.Random(self.seed)  # Set random seed for reproducibility
        user_to_pos_items = {u: list(p) for u, p in self.user_to_pos_items.items()}
        user_pos_to_indices = {k: list(v) for k, v in self.user_pos_to_indices.items()}

        while len(user_pos_to_indices) > 0:
            batch_indices = []
            while len(batch_indices) < self.batch_size and len(user_pos_to_indices) > 0:
                # user with replacement
                user_pool = list({u for u, _ in user_pos_to_indices.keys()})
                user = rnd.choice(user_pool)

                available_pos_items = [
                    p
                    for p in user_to_pos_items[user]
                    if (user, p) in user_pos_to_indices and len(user_pos_to_indices[(user, p)]) > 0
                ]

                # NOTE: this program may fail because the we drop samples here, making unaligned __len__ and __iter__
                if len(available_pos_items) < self.min_pos_items:
                    for pos in available_pos_items:
                        key = (user, pos)
                        if key in user_pos_to_indices and len(user_pos_to_indices[key]) > 0:
                            del user_pos_to_indices[key]  # remove unfulfilled
                    continue

                num_pos_to_take = min(len(available_pos_items), self.max_pos_items)
                sampled_pos_items = available_pos_items[:num_pos_to_take]

                user_triplets = []
                for pos_item in sampled_pos_items:
                    key = (user, pos_item)
                    available_triplets = user_pos_to_indices[key]
                    num_to_take = min(
                        len(available_triplets),
                        self.batch_size - len(batch_indices) - len(user_triplets),  # remaining space in batch
                    )

                    user_triplets.extend(available_triplets[:num_to_take])
                    user_pos_to_indices[key] = available_triplets[num_to_take:]

                    if len(user_pos_to_indices[key]) == 0:
                        del user_pos_to_indices[key]  # remove exhausted

                    if len(user_triplets) >= self.max_triplets_per_user:
                        break

                batch_indices.extend(user_triplets)

            if len(batch_indices) == self.batch_size:
                yield batch_indices
            elif len(batch_indices) < self.batch_size and not self.drop_last:
                yield batch_indices

    def __len__(self):
        # NOTE: this is an approximation, as the actual number of batches may vary due to iterative sampling
        return self.total_instances // self.batch_size - self.drop_last - self.buffer


def get_user_triplet_mapping(
    df: pd.DataFrame, min_pos_items: int = 2
) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], List[int]]]:

    user_to_pos_items = defaultdict(set)
    user_pos_to_indices = defaultdict(list)

    for idx, row in df.iterrows():
        u, p = row["userID"], row["pos_item_id"]
        user_pos_to_indices[(u, p)].append(idx)
        user_to_pos_items[u].add(p)

    # filter users with ≥ min_pos_items positive items
    for user in list(user_to_pos_items.keys()):
        if len(user_to_pos_items[user]) < min_pos_items:
            for pos_item in user_to_pos_items[user]:
                key = (user, pos_item)
                del user_pos_to_indices[key]
            del user_to_pos_items[user]

    return user_to_pos_items, user_pos_to_indices


def validate_unique_pairs(a, b):
    # Stack and find unique (a, b) pairs
    pairs = torch.stack([a, b], dim=1)
    unique_pairs = torch.unique(pairs, dim=0)

    # Count number of unique b's per a
    from collections import defaultdict

    a_to_b = defaultdict(set)
    for a_val, b_val in unique_pairs.tolist():
        a_to_b[a_val].add(b_val)

    # Validate all a have at least 2 unique b's
    all_valid = all(len(b_set) >= 2 for b_set in a_to_b.values())

    print("All a have ≥2 unique b values:", all_valid)
