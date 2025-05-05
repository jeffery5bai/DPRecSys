import os
import random
import sys
from collections import Counter
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# constant
MOVIELENS_DATA_DIR = "../datasets/movielens-2k/user_ratedmovies.dat"
RANDOM_SEED = 42
YEAR_RANGE = (2006, 2008)
RATING_THRESHOLD = 4.0

# TODO: change this to filter out cold-start user/item
MAX_USER_NUM, MAX_ITEM_NUM = None, None
MIN_USER_NUM, MIN_ITEM_NUM = 0, 0
USER_ID_FIELD = "userID"
ITEM_ID_FIELD = "movieID"
YEAR_FIELD = "date_year"
RATING_FIELD = "rating"
TIMESTAMP_FIELD = "timestamp"

# TODO: change this to assign other dir for user/item id mappling
USER_MAPPING_DIR = "userid_mapping.csv"
ITEM_MAPPING_DIR = "itemid_mapping.csv"


"""
Toolkits:
- `set_seed`: Set random seed for reproducibility.
- `DataPreprocessor`: Class for data preprocessing.
    - `load_and_process_df`: Load and process the dataset.
        - `_get_illegal_ids_by_inter_num`: Get illegal ids by interaction number.
        - `_filter_by_threshold`: Filter out user/item with interactions less than min threshold.
    - `stratified_time_split`: Split the dataset into train, validation, and test sets based on user interactions over time.
    - `create_interaction_graph`: Create a user-item interaction graph from the DataFrame.
- `UserItemPairDataset`: Dataset class for user-item pairs.
"""


def set_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducibility.
    Args:
        seed (int): The random seed to set.
    """
    random.seed(seed)
    print("random seed set to", seed)
    np.random.seed(seed)
    print("numpy seed set to", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # make sure the cudnn uses deterministic algorithms
    print("torch seed set to", seed)

    # NOTE: set seed for pytorch-lightning
    seed_everything(seed, workers=True)
    print("lightning seed set to", seed)


# NOTE: Pre-process Dataset


class DataPreprocessor:
    """
    Data Preprocessor for MovieLens dataset.
    """

    def load_and_process_df(
        self,
        file_dir: str = MOVIELENS_DATA_DIR,
        year_range: Tuple[int] = YEAR_RANGE,
        u_mapping_file: str = USER_MAPPING_DIR,
        i_mapping_file: str = ITEM_MAPPING_DIR,
        user_col: str = USER_ID_FIELD,
        item_col: str = ITEM_ID_FIELD,
        year_col: str = YEAR_FIELD,
        rating_col: str = RATING_FIELD,
        max_user_num: int = MAX_USER_NUM,
        max_item_num: int = MAX_ITEM_NUM,
        min_user_num: int = MIN_USER_NUM,
        min_item_num: int = MIN_ITEM_NUM,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Load and process the dataset.
        Args:
            file_dir (str): The path to the dataset file.
            year_range (Tuple[int]): The range of years to filter the data.
            u_mapping_file (str): The path to save the user mapping file.
            i_mapping_file (str): The path to save the item mapping file.
        Returns:
            pd.DataFrame: The processed DataFrame containing user-item interactions.
        """
        # NOTE: load file
        interaction_df = pd.read_table(os.path.join(file_dir))
        print("Data count:", len(interaction_df))

        # NOTE: filter by year and count the cardinalities of user/item
        interaction_df = interaction_df.loc[interaction_df[year_col].between(*year_range), :].reset_index(
            drop=True
        )
        print(f"Data count after filtering by year {year_range}:", len(interaction_df))
        num_user = interaction_df[user_col].nunique()
        num_item = interaction_df[item_col].nunique()
        print("Num of distinct users:", num_user)
        print("Num of distinct items:", num_item)
        print("done!")
        print("---" * 10)

        # NOTE: filter out user/item with interactions less than min threshold
        df = self._filter_by_threshold(
            interaction_df,
            user_col=user_col,
            item_col=item_col,
            max_user_num=max_user_num,
            max_item_num=max_item_num,
            min_user_num=min_user_num,
            min_item_num=min_item_num,
            verbose=verbose,
        )
        print(f"Filtering by min user/item interactions ({min_user_num}/{min_item_num}):")
        print(f"Data count before:", len(interaction_df))
        print("Data count after:", len(df))
        print("done!")
        print("---" * 10)
        # NOTE: Re-index
        uni_users = sorted(pd.unique(df[user_col]))
        uni_items = sorted(pd.unique(df[item_col]))
        # start from 0
        u_id_map = {k: i for i, k in enumerate(uni_users)}
        i_id_map = {k: i for i, k in enumerate(uni_items)}
        df[user_col] = df[user_col].map(u_id_map)
        df[item_col] = df[item_col].map(i_id_map)
        df[user_col] = df[user_col].astype(int)
        df[item_col] = df[item_col].astype(int)
        # dump
        u_df = pd.DataFrame(list(u_id_map.items()), columns=["from", "to"])
        i_df = pd.DataFrame(list(i_id_map.items()), columns=["from", "to"])
        u_df.to_csv(os.path.join(u_mapping_file), index=False)
        i_df.to_csv(os.path.join(i_mapping_file), index=False)
        print(f"Re-index mapping dumped into ...")
        print(f"user: {u_mapping_file}")
        print(f"item: {i_mapping_file}")
        print("done!")
        print("---" * 10)
        # NOTE: get the timestamp / label the target based on the rating threshold
        df["timestamp"] = pd.to_datetime(
            {
                "year": df["date_year"],
                "month": df["date_month"],
                "day": df["date_day"],
                "hour": df["date_hour"],
                "minute": df["date_minute"],
                "second": df["date_second"],
            }
        )
        df["label"] = (df[rating_col] >= RATING_THRESHOLD).astype(int)
        # NOTE: Final Data Info
        print("==== Final Data Info: ====")
        print("Data Year Range:", year_range)
        print("Rating Threshold:", RATING_THRESHOLD)
        print("Num of interactions:", len(df))
        print("Num of distinct users:", df[user_col].nunique())
        print("Num of distinct items:", df[item_col].nunique())
        return df

    def _get_illegal_ids_by_inter_num(
        self,
        df: pd.DataFrame,
        field: str,
        max_num: int = None,
        min_num: int = None,
        verbose: bool = False,
    ) -> Set[int]:
        """
        Get illegal ids by interaction number.
        (helper function)
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            field (str): The column name for user/item IDs.
            max_num (int): The maximum number of interactions allowed.
            min_num (int): The minimum number of interactions allowed.
            verbose (bool): Whether to print the number of illegal ids.
        Returns:
            Set[int]: A set of illegal ids based on the interaction number.
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

    def _filter_by_threshold(
        self,
        raw_df: pd.DataFrame,
        user_col: str,
        item_col: str,
        max_user_num: int,
        max_item_num: int,
        min_user_num: int,
        min_item_num: int,
        verbose: bool,
    ) -> pd.DataFrame:
        """
        Filter out user/item with interactions less than min threshold. (recursively)
        (helper function)
        Args:
            raw_df (pd.DataFrame): The input DataFrame containing user-item interactions.
            verbose (bool): Whether to print the number of illegal ids.
        Returns:
            pd.DataFrame: The filtered DataFrame containing user-item interactions.
        """
        df = raw_df.copy()
        while True:
            ban_users = self._get_illegal_ids_by_inter_num(
                df, field=user_col, max_num=max_user_num, min_num=min_user_num, verbose=verbose
            )
            ban_items = self._get_illegal_ids_by_inter_num(
                df, field=item_col, max_num=max_item_num, min_num=min_item_num, verbose=verbose
            )
            if len(ban_users) == 0 and len(ban_items) == 0:
                return df.reset_index(drop=True)

            dropped_inter = pd.Series(False, index=df.index)
            if user_col:
                dropped_inter |= df[user_col].isin(ban_users)
            if item_col:
                dropped_inter |= df[item_col].isin(ban_items)
            if verbose:
                print(f"{len(dropped_inter)} dropped interactions")
            df.drop(df.index[dropped_inter], inplace=True)

    # NOTE: Train/Val/Test Split
    def stratified_time_split(
        self,
        df: pd.DataFrame,
        user_col: str = USER_ID_FIELD,
        time_col: str = TIMESTAMP_FIELD,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train, validation, and test sets based on user interactions over time.
        Individual-level time-based split.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            user_col (str): The column name for user IDs.
            time_col (str): The column name for timestamps.
            train_ratio (float): The ratio of the training set.
            val_ratio (float): The ratio of the validation set.
            test_ratio (float): The ratio of the test set.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test DataFrames.
        """
        # Assert: ratios must sum to 1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        train_list, val_list, test_list = [], [], []

        for user_id, user_df in df.groupby(user_col):
            user_df = user_df.sort_values(by=time_col).reset_index(drop=True)

            n = len(user_df)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train = user_df.iloc[:n_train]
            val = user_df.iloc[n_train : n_train + n_val]
            test = user_df.iloc[n_train + n_val :]

            train_list.append(train)
            val_list.append(val)
            test_list.append(test)

        df_train = pd.concat(train_list).reset_index(drop=True)
        df_val = pd.concat(val_list).reset_index(drop=True)
        df_test = pd.concat(test_list).reset_index(drop=True)

        total_cnt = len(df)
        print(
            f"Splitting data into train/valid/test by time period with ratio=({train_ratio} : {val_ratio} : {test_ratio}):"
        )
        print(f"train: {len(df_train)} ({round(len(df_train) / total_cnt * 100, 2)}%")
        print(f"valid: {len(df_val)} ({round(len(df_val) / total_cnt * 100, 2)}%)")
        print(f"test: {len(df_test)} ({round(len(df_test) / total_cnt * 100, 2)}%)")
        print("---" * 10, "\n")
        print("Check target label distribution after splitting (%):")
        print("train", df_train["label"].value_counts(normalize=True))
        print("valid", df_val["label"].value_counts(normalize=True))
        print("test", df_test["label"].value_counts(normalize=True))

        return df_train, df_val, df_test

    # NOTE: User-Item Bi-partite Graph Construction
    def create_interaction_graph(self, df_split: pd.DataFrame) -> Data:
        """
        Create a user-item interaction graph from the DataFrame.
        Args:
            df_split (pd.DataFrame): The input DataFrame containing user-item interactions.
        Returns:
            Data: A PyTorch Geometric Data object representing the user-item interaction graph.
        """
        print("Creating interaction graph...")
        print("Drop negative samples")
        print("  Num of all interactions:", len(df_split))
        df_split = df_split.loc[df_split["label"] == 1, :]
        print("  Num of positive interactions:", len(df_split), "\n")

        print("Building edges...")
        edge_index = torch.tensor(
            np.array([df_split["userID"].values, df_split["movieID"].values]), dtype=torch.long
        )
        print("Building bi-directed edges (duplications)...")
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)  # [2, num_edges*2]
        print("Building labels...")
        edge_label = torch.tensor(df_split["label"].values, dtype=torch.float)
        print("Building bi-directed labels (duplications)...", "\n")
        edge_label = torch.cat((edge_label, edge_label))
        data = Data(edge_index=edge_index, edge_label=edge_label)
        print("Interaction Graph:", data)
        print("Edge Index:", data.edge_index)

        return data


class UserItemPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        User-Item Pair Dataset for training and evaluation.
        Args:
            df (pd.DataFrame): with columns ['user_id', 'item_id', 'label']
        """
        self.user_ids = torch.tensor(df['userID'].values, dtype=torch.long)
        self.item_ids = torch.tensor(df['movieID'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.tensor]:
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]
