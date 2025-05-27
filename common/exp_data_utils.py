import os
import random
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

# constant
MOVIELENS_DATA_DIR = "../datasets/hetrec2011-movielens-2k-v2/user_ratedmovies.dat"
ACTOR_DATA_DIR = "../datasets/hetrec2011-movielens-2k-v2/movie_actors.dat"
COUNTRY_DATA_DIR = "../datasets/hetrec2011-movielens-2k-v2/movie_countries.dat"
DIRECTOR_DATA_DIR = "../datasets/hetrec2011-movielens-2k-v2/movie_directors.dat"
GENRE_DATA_DIR = "../datasets/hetrec2011-movielens-2k-v2/movie_genres.dat"

RANDOM_SEED = 42
YEAR_RANGE = (2006, 2008)
RATING_THRESHOLD = 4.0

# TODO: change this to filter out cold-start user/item
MAX_USER_NUM, MAX_ITEM_NUM = None, None
MIN_USER_NUM, MIN_ITEM_NUM = 0, 0
USER_ID_FIELD = "userID"
ITEM_ID_FIELD = "movieID"
POS_ITEM_FIELD = "pos_item"
NEG_ITEM_FIELD = "neg_item"
YEAR_FIELD = "date_year"
TIMESTAMP_FIELD = "timestamp"
RATING_FIELD = "rating"
LABEL_FIELD = "label"
FEATURE_FIELD = ["actorID", "country", "directorID", "genre"]
FEATURE_IDX_FIELD = ["actorID_idx", "country_idx", "directorID_idx", "genre_idx"]
IS_LIST_FEATURES = [True, False, False, True]


class ExperimentDataPreprocessor:
    """
    Data preprocessor for experiment preparation.
    """

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
        print("train", df_train[LABEL_FIELD].value_counts(normalize=True))
        print("valid", df_val[LABEL_FIELD].value_counts(normalize=True))
        print("test", df_test[LABEL_FIELD].value_counts(normalize=True))

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
            np.array([df_split[USER_ID_FIELD].values, df_split[ITEM_ID_FIELD].values]), dtype=torch.long
        )
        print("Building labels...")
        edge_label = torch.tensor(df_split[LABEL_FIELD].values, dtype=torch.float)
        data = Data(edge_index=edge_index, edge_label=edge_label)
        print("Interaction Graph:", data)
        print("Edge Index:", data.edge_index)

        return data

    def prepare_triplet_df(
        self, df_with_encoded_features: pd.DataFrame, k_negative_samples: int = 5, seed: int = RANDOM_SEED
    ) -> pd.DataFrame:
        """
        Prepare the DataFrame for triplet-based training.
        Args:
            df_with_encoded_features (pd.DataFrame): The DataFrame containing user-item interactions with encoded features.
        Returns:
            pd.DataFrame: A DataFrame containing user-item pairs with ratings and encoded features.
        """
        # NOTE: Reserve item features and join after sampling
        item_feature_df = (
            df_with_encoded_features[[ITEM_ID_FIELD, *FEATURE_IDX_FIELD]]
            .drop_duplicates([ITEM_ID_FIELD], ignore_index=True)
            .copy()
        )
        all_items = item_feature_df[ITEM_ID_FIELD].unique()

        # NOTE: Only keep positive samples for triplet-based training (implicit feedback)
        df = (
            df_with_encoded_features.loc[
                df_with_encoded_features[LABEL_FIELD] == 1, [USER_ID_FIELD, ITEM_ID_FIELD]
            ]
            .rename(columns={ITEM_ID_FIELD: POS_ITEM_FIELD})
            .copy()
        )

        result_dfs = []
        np.random.seed(seed)
        for user_id, user_df in df.groupby(USER_ID_FIELD):
            user_items = user_df[POS_ITEM_FIELD].unique()
            unseen_items = np.setdiff1d(all_items, user_items)
            negative_items = np.random.choice(unseen_items, size=k_negative_samples, replace=False)

            negative_df = pd.DataFrame({USER_ID_FIELD: user_id, NEG_ITEM_FIELD: negative_items})
            sampled_df = user_df.merge(negative_df, on=[USER_ID_FIELD], how="left")

            result_dfs.append(sampled_df)

        triplet_df = pd.concat(result_dfs, ignore_index=True)
        triplet_df = triplet_df.merge(
            item_feature_df, left_on=POS_ITEM_FIELD, right_on=ITEM_ID_FIELD, how="inner"
        )
        print(f"Original data count (positive samples): {len(df)}")
        print(
            f"Num of triplets: {len(df)}(pos samples) * {k_negative_samples}(negative sampled items) = {len(triplet_df)}"
        )
        return triplet_df

    def prepare_prediction_df(self, df: pd.DataFrame, K: int = 500, seed=RANDOM_SEED) -> pd.DataFrame:
        """
        Prepare the prediction DataFrame for the test set.
        Performs negative sampling for users who have less than K interactions.
        Args:
            df (pd.DataFrame): The DataFrame containing user-item interactions.
            K (int): The number of items to sample for each user.
            seed (int): The random seed for reproducibility.
        Returns:
            pd.DataFrame: A DataFrame containing user-item pairs with ratings.
        """
        # NOTE: Reserve item features and join after sampling
        item_feature_df = (
            df[[ITEM_ID_FIELD, *FEATURE_IDX_FIELD]].drop_duplicates([ITEM_ID_FIELD], ignore_index=True).copy()
        )

        df = df.loc[:, [USER_ID_FIELD, ITEM_ID_FIELD, LABEL_FIELD]].copy()
        n_users = df[USER_ID_FIELD].nunique()
        n_items = df[ITEM_ID_FIELD].nunique()
        all_items = df[ITEM_ID_FIELD].unique()

        result_dfs = []
        np.random.seed(seed)
        for user_id, user_df in df.groupby(USER_ID_FIELD):
            user_items = user_df[ITEM_ID_FIELD].unique()

            num_existing = len(user_df)
            if num_existing >= K:
                sampled_df = user_df.sample(n=K, random_state=seed)
            else:
                num_to_sample = K - num_existing
                unseen_items = np.setdiff1d(all_items, user_items)

                # NOTE: Sample unseen items
                sampled_items = np.random.choice(unseen_items, size=num_to_sample, replace=False)

                # NOTE: Create negative samples with dummy values
                negative_df = pd.DataFrame(
                    {
                        USER_ID_FIELD: user_id,
                        ITEM_ID_FIELD: sampled_items,
                        LABEL_FIELD: 0,
                    }
                )

                sampled_df = pd.concat([user_df, negative_df], ignore_index=True)

            result_dfs.append(sampled_df)

        prediction_df = pd.concat(result_dfs, ignore_index=True)
        prediction_df = prediction_df.merge(item_feature_df, on=ITEM_ID_FIELD, how="inner")
        print("Prediction DataFrame:")
        print(f"User Pool: {n_users}")
        print(f"Item Pool: {n_items}, negative sampled to {K} items for each user")
        print(f"Num of interactions: {n_users}(users) * {K}(items) = {len(prediction_df)}")
        return prediction_df
