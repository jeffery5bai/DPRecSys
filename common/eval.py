import gc
import os
import sys
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from common.utils import DataPreprocessor, FeatureEngineer
from pytorch_lightning import seed_everything
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
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

OOV_TOKEN = "[OOV]"


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
        rescale: bool = True,
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
        for i, (user_id, df) in enumerate(
            tqdm(user_groups, desc="Calculating user diversity preference scale")
        ):
            user_profile = {USER_ID_FIELD: user_id}
            ratings = df[RATING_FIELD].values

            for feat, feat_idx in zip(FEATURE_FIELD, FEATURE_IDX_FIELD):
                # NOTE: transform the feature columns to multi-hot encoding
                feature_indices = df[feat_idx].values
                cardinality = len(feature_vocab2idx[feat])
                multihots = np.stack(
                    [self._indices_to_multi_hot(indices, cardinality) for indices in feature_indices]
                )

                # NOTE: calculate the weighted average of the multi-hot vectors
                weighted_vec = (multihots * ratings[:, np.newaxis]).sum(axis=0)

                user_profile[f"{feat}_wvec"] = weighted_vec
                user_profile[f"{feat}_dps"] = self.entropy(weighted_vec, normalized=normalized)
                del multihots, weighted_vec
            if i % 250 == 0:
                gc.collect()

            user_profiles.append(user_profile)

        # NOTE: Re-scale the DPS values exactly to [0, 1]
        df = pd.DataFrame(user_profiles)
        if rescale:
            rescale_features = [f"{feat}_dps" for feat in FEATURE_FIELD]
            scaler = MinMaxScaler()
            df[rescale_features] = scaler.fit_transform(df[rescale_features])

        return df

    def _indices_to_multi_hot(self, indices: Union[int, List[int]], cardinality: int) -> np.ndarray:
        vec = np.zeros(cardinality, dtype=np.int32)
        vec[indices] = 1
        vec = vec[2:]  # to skip the first and index (0, 1) which is reserved for padding and rare tokens
        return vec

    def entropy(self, vec, normalized=True):
        """Calculate the Shannon Entropy of a tensor vector."""
        vec_sum = np.sum(vec)
        if vec_sum == 0:
            return np.nan  # undefined

        p = vec / (vec_sum + 1e-8)  # normalize weighted vec to probability distribution
        e = -np.sum(p[p > 0] * np.log(p[p > 0]))  # filter out zero entries to avoid log(0)
        return (e / np.log(len(p))) if normalized else e  # normalize to [0, 1]

    def get_item_feature_multihot_vec(
        self, df_with_encoded_features: pd.DataFrame, feature_vocab2idx: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Transform the item features in df_with_encoded_features to multi-hot encoding vectors.
        Args:
            df_with_encoded_features (pd.DataFrame): DataFrame with item features and their encoded indices.
                It should contain columns for item ID, feature fields, and their corresponding indices.
            feature_vocab2idx (Dict[str, int]): Dictionary mapping feature names to their vocabulary size.
        Returns:
            pd.DataFrame: DataFrame with item IDs and their corresponding multi-hot encoded feature vectors.
                The columns will be [ITEM_ID_FIELD, f"{feat}_vec"] for each feature in FEATURE_FIELD.
        """
        item_feature_vec_df = (
            df_with_encoded_features[[ITEM_ID_FIELD] + FEATURE_IDX_FIELD]
            .drop_duplicates(subset=ITEM_ID_FIELD, keep="first")
            .reset_index(drop=True)
        ).copy()

        for feat, feat_idx in zip(FEATURE_FIELD, FEATURE_IDX_FIELD):
            # NOTE: transform the feature columns to multi-hot encoding
            feature_indices = item_feature_vec_df[feat_idx].values
            item_feature_vec_df[f"{feat}_vec"] = [
                self._indices_to_multi_hot(indices, len(feature_vocab2idx[feat]))
                for indices in feature_indices
            ]
        return item_feature_vec_df[[ITEM_ID_FIELD] + [f"{feat}_vec" for feat in FEATURE_FIELD]]

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

    def process_eval_df(
        self,
        eval_df: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        k: int = 100,
        actor_k: int = 5,
        rare_threshold: int = 5,
        encoded: bool = True,
    ) -> pd.DataFrame:
        """
        Process the evaluation DataFrame to ensure it has the correct structure.
        Args:
            eval_df (pd.DataFrame): DataFrame with columns ["user", "rec_items", "gt_items"].
            feature_engineer (FeatureEngineer): Feature engineer instance to transform item features.
            k (int): Number of top items to consider for recommendations.
            actor_k (int): Number of actors to consider for each item.
            rare_threshold (int): Threshold for filtering out rare items.

        Returns:
            pd.DataFrame: Processed DataFrame with encoded features ready for evaluation.
        """
        # NOTE: Explode items and slice to top K
        df = eval_df.copy()
        df["topk_items"] = df["rec_items"].apply(lambda x: x[:k])
        print("candidate item pool size:", k)
        exploded = df.explode("topk_items")
        exploded = exploded.rename(columns={"user": USER_ID_FIELD, "topk_items": ITEM_ID_FIELD})
        exploded["rank"] = exploded.groupby(USER_ID_FIELD).cumcount() + 1
        exploded = exploded[exploded[ITEM_ID_FIELD] != OOV_TOKEN]
        exploded[ITEM_ID_FIELD] = exploded[ITEM_ID_FIELD].astype(int)
        exploded = exploded[[USER_ID_FIELD, ITEM_ID_FIELD, "rank"]]
        print("exploded", exploded[USER_ID_FIELD].nunique())

        # NOTE: Join item info
        exploded = DataPreprocessor().join_item_features(
            exploded,
            actor_k=actor_k,
            threshold=rare_threshold,
        )

        # NOTE: Encode features
        if encoded:
            exploded = feature_engineer.transform(exploded)
            print("encoded", exploded[USER_ID_FIELD].nunique())

        return exploded

    def evaluate_dpms_at_k(
        self,
        eval_df: pd.DataFrame,
        feature_engineer: FeatureEngineer,
        ground_truth_dps_df: pd.DataFrame,
        k: int = 5,
        actor_k: int = 5,
        rare_threshold: int = 5,
    ) -> pd.DataFrame:
        """
        Evaluate the diversity preference matching score (DPMS) at K.
        eval_df: DataFrame with columns ["user", "rec_items", "gt_items"].
        """
        encoded_df = self.process_eval_df(
            eval_df, feature_engineer, k=k, actor_k=actor_k, rare_threshold=rare_threshold
        )

        # NOTE: Calculate Prediction DP vectors for each user
        encoded_df["rating"] = 1.0  # Set a dummy rating for the purpose of DPS calculation
        pred_user_dps_df = self.eval_user_diversity_preference_scale(
            df_with_encoded_features=encoded_df, feature_vocab2idx=feature_engineer.vocab2idx, normalized=True
        )
        pred_user_dps_df = pred_user_dps_df.rename(
            columns={f"{feat}_wvec": f"{feat}_pred" for feat in FEATURE_FIELD},
        )
        pred_user_dps_df = pred_user_dps_df[[USER_ID_FIELD] + [f"{feat}_pred" for feat in FEATURE_FIELD]]

        # NOTE: Join pred and ground truth DP vectors
        ground_truth_dps_df = ground_truth_dps_df.rename(
            columns={f"{feat}_wvec": f"{feat}_gt" for feat in FEATURE_FIELD},
        )
        ground_truth_dps_df = ground_truth_dps_df[[USER_ID_FIELD] + [f"{feat}_gt" for feat in FEATURE_FIELD]]

        combined_df = pred_user_dps_df.merge(
            ground_truth_dps_df,
            on=USER_ID_FIELD,
            how="inner",
        )

        print("combined_df", combined_df[USER_ID_FIELD].nunique())

        # NOTE: Calculate DPMS
        print("Calculating DPMS for each feature...")
        for feat in FEATURE_FIELD:
            combined_df[f"{feat}_dpms"] = cosine_similarity(
                np.stack(combined_df[f"{feat}_pred"].values), np.stack(combined_df[f"{feat}_gt"].values)
            ).diagonal()
        combined_df["avg_dpms"] = combined_df[[f"{feat}_dpms" for feat in FEATURE_FIELD]].mean(axis=1)

        return combined_df[[USER_ID_FIELD] + [f"{feat}_dpms" for feat in FEATURE_FIELD] + ["avg_dpms"]]

    def evaluate_ils_at_k(
        self,
        encoded_eval_df: pd.DataFrame,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Evaluate the intra-list similarity (ILS) at K.
        Args:
            eval_df: DataFrame with columns ["user", "rec_items", "gt_items"]. (item ids MUST BE encoded)
            sim_matrix: Precomputed item-item similarity matrix.
            k: Number of top items to consider for ILS calculation.
        returns:
            pd.DataFrame: DataFrame with columns ["user", f"ILS@{k}"].
        """
        # load files
        data = np.load("../experiments/artifacts/item_sim_data.npz", allow_pickle=True)
        sim_matrix = data["sim_matrix"]
        movie2idx = data["movie2idx"].item()

        encoded_eval_df[f"ILS@{k}"] = encoded_eval_df["rec_items"].apply(
            lambda x: self._avg_pairwise_similarity(sim_matrix, x[:k], movie2idx)
        )
        return encoded_eval_df[["user", f"ILS@{k}"]]

    def _avg_pairwise_similarity(
        self, sim_matrix: np.ndarray, item_indices: np.ndarray, movie2idx: dict
    ) -> float:
        """
        Compute average pairwise similarity among a set of items.

        Parameters:
            sim_matrix: 2D NumPy array of shape (n_items, n_items)
            item_indices: 1D array of indices (integers)

        Returns:
            Average pairwise similarity (float)
        """

        # Map the raw movieID to valid indices in the sim_matrix
        item_indices = [movie2idx[item] for item in item_indices if item in movie2idx]

        if len(item_indices) < 2:
            return 0.0  # or np.nan, since no pair exists

        # Extract submatrix
        submatrix = sim_matrix[np.ix_(item_indices, item_indices)]

        # Exclude diagonal (self-similarity)
        triu_indices = np.triu_indices(len(item_indices), k=1)
        pairwise_sims = submatrix[triu_indices]

        return pairwise_sims.mean()
