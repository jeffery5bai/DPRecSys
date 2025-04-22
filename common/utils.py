import os
import random
import sys
from collections import Counter
from typing import Dict, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# constant
MOVIELENS_DATA_DIR = "../datasets/movielens-2k/user_ratedmovies.dat"
RANDOM_SEED = 42
YEAR_RANGE = (2006, 2008)

# TODO: change this to filter out cold-start user/item
MIN_USER_NUM, MIN_ITEM_NUM = 0, 0
USER_ID_FIELD = "userID"
ITEM_ID_FIELD = "movieID"

# TODO: change this to assign other dir for user/item id mappling
USER_MAPPING_DIR = 'userid_mapping.csv'
ITEM_MAPPING_DIR = 'itemid_mapping.csv'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # make sure the cudnn uses deterministic algorithms


# NOTE: Pre-process Dataset
def _get_illegal_ids_by_inter_num(
    df: pd.DataFrame,
    field: str,
    max_num: int = None,
    min_num: int = None,
    verbose: bool = False,
) -> Set[int]:
    """
    Get user/item ids with interactions less or more than the thresholds.
    (helper function)
    """
    if field is None:
        return set()
    if max_num is None and min_num is None:
        return set()

    max_num = max_num or np.inf
    min_num = min_num or -1

    ids = df[field].values
    inter_num = Counter(ids)
    ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
    if verbose:
        print(f"{len(ids)} illegal_ids_by_inter_num, field={field}")

    return ids


def _filter_by_k_core(
    raw_df: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filter out all illegal ids.
    (helper function)
    """
    df = raw_df.copy()
    while True:
        ban_users = _get_illegal_ids_by_inter_num(
            df, field=USER_ID_FIELD, max_num=None, min_num=MIN_USER_NUM, verbose=verbose
        )
        ban_items = _get_illegal_ids_by_inter_num(
            df, field=ITEM_ID_FIELD, max_num=None, min_num=MIN_ITEM_NUM, verbose=verbose
        )
        if len(ban_users) == 0 and len(ban_items) == 0:
            print("done!")
            return df.reset_index(drop=True)

        dropped_inter = pd.Series(False, index=df.index)
        if USER_ID_FIELD:
            dropped_inter |= df[USER_ID_FIELD].isin(ban_users)
        if ITEM_ID_FIELD:
            dropped_inter |= df[ITEM_ID_FIELD].isin(ban_items)
        if verbose:
            print(f"{len(dropped_inter)} dropped interactions")
        df.drop(df.index[dropped_inter], inplace=True)


def load_and_process_df(
    file_dir: str = MOVIELENS_DATA_DIR,
    year_range: Tuple[int] = YEAR_RANGE,
    u_mapping_file: str = USER_MAPPING_DIR,
    i_mapping_file: str = ITEM_MAPPING_DIR,
) -> pd.DataFrame:
    """
    Load, filter and re-index MovieLens-2k interaction data.
    """
    # NOTE: load file
    interaction_df = pd.read_table(os.path.join(file_dir))
    print("Data count:", len(interaction_df))
    # interaction_df.head()

    # NOTE: filter by year and count the cardinalities of user/item
    interaction_df = interaction_df.loc[interaction_df["date_year"].between(*year_range), :].reset_index(
        drop=True
    )
    print(f"Data count after filtering by year {YEAR_RANGE}:", len(interaction_df))
    NUM_USER = interaction_df["userID"].nunique()
    NUM_ITEM = interaction_df["movieID"].nunique()
    print("Num of distinct users:", NUM_USER)
    print("Num of distinct items:", NUM_ITEM)
    print("---"*10)

    # NOTE: filter out user/item with interactions less than min threshold
    df = _filter_by_k_core(interaction_df, verbose=True)
    print(f"Filtering by min user/item interactions ({MIN_USER_NUM}/{MIN_ITEM_NUM}):")
    print(f"Data count before:", len(interaction_df))
    print("Data count after:", len(df))
    print("---"*10)

    # NOTE: Re-index
    uni_users = sorted(pd.unique(df[USER_ID_FIELD]))
    uni_items = sorted(pd.unique(df[ITEM_ID_FIELD]))

    # start from 0
    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}

    df[USER_ID_FIELD] = df[USER_ID_FIELD].map(u_id_map)
    df[ITEM_ID_FIELD] = df[ITEM_ID_FIELD].map(i_id_map)
    df[USER_ID_FIELD] = df[USER_ID_FIELD].astype(int)
    df[ITEM_ID_FIELD] = df[ITEM_ID_FIELD].astype(int)

    # dump
    u_df = pd.DataFrame(list(u_id_map.items()), columns=['from', 'to'])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=['from', 'to'])

    u_df.to_csv(os.path.join(u_mapping_file), index=False)
    i_df.to_csv(os.path.join(i_mapping_file), index=False)
    print(f'mapping dumped into ...')
    print(f"user: {USER_MAPPING_DIR}")
    print(f"item: {ITEM_MAPPING_DIR}")

    return df







