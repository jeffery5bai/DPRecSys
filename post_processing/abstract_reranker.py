import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from common.eval import Evaluator
from common.utils import DataPreprocessor, FeatureEngineer


class Reranker:
    """
    Abstract base class for rerankers.
    All rerankers should inherit from this class and implement the `rerank` method.
    """

    def __init__(self, eval_df: pd.DataFrame, feature_engineer: FeatureEngineer, m: int = 100):
        """
        Initialize the reranker with a configuration dictionary.

        Args:
            eval_df (pd.DataFrame): DataFrame containing evaluation data with columns ["user", "rec_items", "gt_items"].
            feature_engineer (FeatureEngineer): Instance of FeatureEngineer for feature processing.
            m (int): Number of items in candidate pool to consider for reranking.
        """
        self.eval_df = eval_df
        self.feature_engineer = feature_engineer
        self.m = m

        self.user_id_field = "userID"
        self.item_id_field = "movieID"
        self.feature_fields = ["actorID", "country", "directorID", "genre"]
        self.feature_id_fields = ["actorID_idx", "country_idx", "directorID_idx", "genre_idx"]
        self.oov_token = "[OOV]"

        self.evaluator = Evaluator()

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the input DataFrame before reranking.

        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for reranking.
        """
        encoded_df = self.evaluator.process_eval_df(
            eval_df=self.eval_df, feature_engineer=self.feature_engineer, k=self.m
        )
        return encoded_df

    def rerank(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this method.")
