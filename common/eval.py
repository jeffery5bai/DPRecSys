import os
import random
import sys
from collections import Counter
from typing import Dict, List, Set, Tuple, Union

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


def eval_user_diversity_preference_scale(
    df_with_side_info: pd.DataFrame,
    feature_vocab2idx: Dict[str, int],
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Calculate the diversity preference scale (DPS) for each user based on their ratings and side information.
    Args:
        df_with_side_info (pd.DataFrame): DataFrame with user ratings and side information.
            It should contain columns for user ID, item ID, rating, and encoded features.
        feature_vocab2idx (Dict[str, int]): Dictionary mapping feature names to their vocabulary size.
        normalized (bool): Whether to normalize the entropy values to [0, 1].
    Returns:
        pd.DataFrame: DataFrame with user IDs and their corresponding diversity preference scale (DPS) for each feature.
    """

    def _indices_to_multi_hot(indices: Union[int, List[int]], cardinality: int) -> np.ndarray:
        vec = np.zeros(cardinality, dtype=int)
        vec[indices] = 1
        vec = vec[1:]  # to skip the first index (0) which is reserved for padding
        return vec

    # NOTE: calculate the diversity preference scale
    user_groups = df_with_side_info.groupby(USER_ID_FIELD)
    user_profiles = []
    for user_id, df in tqdm(user_groups, desc="Calculating user diversity preference scale"):
        user_profile = {USER_ID_FIELD: user_id}
        ratings = df["rating"].values

        for feat, feat_idx in zip(FEATURE_FIELD, FEATURE_IDX_FIELD):
            # NOTE: transform the feature columns to multi-hot encoding
            df[f"{feat}_vec"] = df[feat_idx].apply(
                lambda x: _indices_to_multi_hot(x, len(feature_vocab2idx[feat]))
            )

            # NOTE: calculate the weighted average of the multi-hot vectors
            multihots = np.stack(df[f"{feat}_vec"].values)
            weighted_vec = (multihots * ratings[:, np.newaxis]).sum(axis=0)

            user_profile[f"{feat}_dist"] = weighted_vec
            user_profile[f"{feat}_dps"] = entropy(weighted_vec, normalized=normalized)

        user_profiles.append(user_profile)

    return pd.DataFrame(user_profiles)


def entropy(vec, normalized=True):
    """Calculate the Shannon Entropy of a vector."""
    vec_sum = np.sum(vec)
    if vec_sum == 0:
        return np.nan  # undefined

    p = vec / (vec_sum + 1e-8)  # normalize weighted vec to probability distribution
    e = -np.sum(p[p > 0] * np.log(p[p > 0]))  # filter out zero entries to avoid log(0)
    return (e / np.log(len(p))) if normalized else e  # normalize to [0, 1]
