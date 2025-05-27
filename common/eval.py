import gc
import os
import random
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from torch_geometric.data import Data
from tqdm import tqdm

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
YEAR_FIELD = "date_year"
TIMESTAMP_FIELD = "timestamp"
RATING_FIELD = "rating"
LABEL_FIELD = "label"
FEATURE_FIELD = ["actorID", "country", "directorID", "genre"]
FEATURE_IDX_FIELD = ["actorID_idx", "country_idx", "directorID_idx", "genre_idx"]
IS_LIST_FEATURES = [True, False, False, True]


class Evaluator:
    def __init__(self, seed: int = RANDOM_SEED):
        """
        Initialize the Evaluator with a random seed for reproducibility.
        Args:
            seed (int): Random seed for reproducibility.
        """
        self.seed = seed
        seed_everything(seed)

    def eval_user_diversity_preference_scale(
        self,
        df_with_encoded_features: pd.DataFrame,
        feature_vocab2idx: Dict[str, int],
        normalized: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the diversity preference scale (DPS) for each user based on their ratings and side information.
        Args:
            df_with_encoded_features (pd.DataFrame): DataFrame with user ratings and side information.
                It should contain columns for user ID, item ID, rating, and encoded features.
            feature_vocab2idx (Dict[str, int]): Dictionary mapping feature names to their vocabulary size.
            normalized (bool): Whether to normalize the entropy values to [0, 1].
        Returns:
            pd.DataFrame: DataFrame with user IDs and their corresponding diversity preference scale (DPS) for each feature.
        """

        # NOTE: calculate the diversity preference scale
        user_groups = df_with_encoded_features.groupby(USER_ID_FIELD)
        user_profiles = []
        for i, (user_id, df) in enumerate(tqdm(user_groups, desc="Calculating user diversity preference scale")):
            user_profile = {USER_ID_FIELD: user_id}
            ratings = df["rating"].values

            for feat, feat_idx in zip(FEATURE_FIELD, FEATURE_IDX_FIELD):
                # NOTE: transform the feature columns to multi-hot encoding
                feature_indices = df[feat_idx].values
                multihots = np.array(
                    [
                        self._indices_to_multi_hot(indices, len(feature_vocab2idx[feat]))
                        for indices in feature_indices
                    ]
                )

                # NOTE: calculate the weighted average of the multi-hot vectors
                weighted_vec = (multihots * ratings[:, np.newaxis]).sum(axis=0)

                user_profile[f"{feat}_wvec"] = weighted_vec
                user_profile[f"{feat}_dps"] = self.entropy(weighted_vec, normalized=normalized)
                del multihots, weighted_vec
            if i % 250 == 0:
                gc.collect()

            user_profiles.append(user_profile)

        return pd.DataFrame(user_profiles)

    def _indices_to_multi_hot(self, indices: Union[int, List[int]], cardinality: int) -> np.ndarray:
        vec = np.zeros(cardinality, dtype=int)
        vec[indices] = 1
        vec = vec[1:]  # to skip the first index (0) which is reserved for padding
        return vec

    def entropy(self, vec, normalized=True):
        """Calculate the Shannon Entropy of a vector."""
        vec_sum = np.sum(vec)
        if vec_sum == 0:
            return np.nan  # undefined

        p = vec / (vec_sum + 1e-8)  # normalize weighted vec to probability distribution
        e = -np.sum(p[p > 0] * np.log(p[p > 0]))  # filter out zero entries to avoid log(0)
        return (e / np.log(len(p))) if normalized else e  # normalize to [0, 1]

    def prepare_evaluation_data(
        self,
        test_results_log: Dict[str, Union[torch.tensor, Any]],
        idx2vocab: Dict[str, Dict[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Restructure the prediction log into a DataFrame for evaluation.
        output schema: ["user", "rec_items", "gt_items"]
        Args:
            test_results_log (Dict[str, Union[torch.tensor, Any]]): Dictionary containing the test results.
            idx2vocab (Dict[str, Dict[int, int]], optional): Mapping from index to original values for user and item IDs.
        Returns:
            pd.DataFrame: DataFrame with columns ["user", "rec_items", "gt_items"].
                - "user": User ID
                - "rec_items": List of recommended item IDs
                - "gt_items": List of ground truth item IDs
        """
        selected_keys = ["user", "item", "score", "label"]
        df = pd.DataFrame({key: test_results_log[key] for key in selected_keys})

        if idx2vocab is not None:
            # Map user and item IDs to their original values using idx2vocab
            df["user"] = df["user"].apply(lambda x: idx2vocab[USER_ID_FIELD][x])
            df["item"] = df["item"].apply(lambda x: idx2vocab[ITEM_ID_FIELD][x])

        def _process_group(group: pd.DataFrame):
            rec_items = group.sort_values("score", ascending=False)["item"].tolist()
            gt_items = group[group["label"] == 1]["item"].tolist()
            return pd.Series({"rec_items": rec_items, "gt_items": gt_items})

        # apply the group transformation
        return df.groupby("user").apply(_process_group, include_groups=False).reset_index()

    def evaluate(self, eval_df: pd.DataFrame, K: int = 5) -> pd.DataFrame:
        """
        Evaluate the recommendations in eval_df using NDCG, recall, and precision at K.
        Args:
            eval_df (pd.DataFrame): DataFrame with columns ["user", "rec_items", "gt_items"].
                - "user": User ID
                - "rec_items": List of recommended item IDs
                - "gt_items": List of ground truth item IDs
            K (int): The number of top recommendations to consider for evaluation.
        Returns:
            pd.DataFrame: The input eval_df with additional columns for NDCG, recall, and precision at K.
        """
        eval_df[f"ndcg@{K}"] = eval_df.apply(
            lambda row: self._ndcg_at_k(row["rec_items"], row["gt_items"], k=K), axis=1
        )
        eval_df[f"recall@{K}"] = eval_df.apply(
            lambda row: self._recall_at_k(row["rec_items"], row["gt_items"], k=K), axis=1
        )
        eval_df[f"precision@{K}"] = eval_df.apply(
            lambda row: self._precision_at_k(row["rec_items"], row["gt_items"], k=K), axis=1
        )
        return eval_df

    def _ndcg_at_k(self, rec_items, gt_items, k=5):
        """
        Calculate the Normalized Discounted Cumulative Gain (NDCG) at k. (for single user)
        Returns:
            float: The NDCG score at k, normalized to [0, 1].
        """

        def dcg(relevance_scores):
            return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))

        relevance = [1 if item in gt_items else 0 for item in rec_items[:k]]
        ideal_relevance = sorted(relevance, reverse=True)

        dcg_score = dcg(relevance)
        idcg_score = dcg(ideal_relevance)
        return float(dcg_score / idcg_score) if idcg_score > 0 else 0.0

    def _recall_at_k(self, rec_items, gt_items, k=5):
        """
        Calculate the recall at k. (for single user)
        Returns:
            float: The recall score at k, normalized to [0, 1].
        """
        hit_set = set(rec_items[:k]) & set(gt_items)
        return len(hit_set) / len(gt_items) if gt_items else 0.0

    def _precision_at_k(self, rec_items, gt_items, k=5):
        """
        Calculate the precision at k. (for single user)
        Returns:
            float: The precision score at k, normalized to [0, 1].
        """
        hit_set = set(rec_items[:k]) & set(gt_items)
        return len(hit_set) / k
