"""
adaptiveMMR (Maximal Marginal Relevance) Reranker.
Vyas, L. K., & Boratto, L. (2025, June). Addressing Personalized Diversity in Eyewear Recommendation: a Lenskart Case Study. 
In Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization (pp. 263-267).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
import pandas as pd
from post_processing.abstract_reranker import Reranker
from scipy.stats import rankdata
from tqdm import tqdm


class AdaptiveMMR(Reranker):
    """
    adaptiveMMR (Maximal Marginal Relevance) Reranker.
    This class implements the MMR algorithm + GS-scores to adaptively rerank items based on diversity of item attributes.
    """

    def __init__(self, user_type: dict, **kwargs):
        """ """
        super().__init__(**kwargs)
        self.user_type = user_type
        # load files
        data = np.load("../experiments/artifacts/item_sim_data.npz", allow_pickle=True)
        self.item_matrix: np.array = data["sim_matrix"]
        self.movie2idx: dict = data["movie2idx"].item()
        self.idx2movie: dict = {v: k for k, v in self.movie2idx.items()}  # reverse mapping

    def rerank(self, top_k: int = 20, theta_s: float = 0.8, theta_g: float = 0.5) -> pd.DataFrame:
        """
        Rerank items for each user based on MMR algorithm.
        Args:
            top_k (int): Number of items to select for each user.
            theta_s (float): Trade-off parameter between relevance and diversity for Specialist.
            theta_g (float): Trade-off parameter between relevance and diversity for Generalist.
        Returns:
            pd.DataFrame: DataFrame containing reranked items for each user.
        """

        # NOTE: this dataframe contains all necessary features
        print("Preparing input DataFrame for adaptive MMR...")
        df = self.preprocess(encoded=False)
        # NOTE: map raw movieID to valid indices
        df[self.item_id_field] = df[self.item_id_field].map(self.movie2idx)

        results = []
        print("Reranking items for each user...")
        for user_id, user_df in tqdm(df.groupby(self.user_id_field), desc="Reranking users"):
            # initialize the selected item
            selected_items = [user_df.loc[user_df["rank"] == 1, self.item_id_field].values[0]]
            # adaptive theta based on user type
            theta = theta_s if user_id in self.user_type["specialist"] else theta_g
            for _ in range(1, top_k):
                candidate_items = user_df.loc[
                    ~user_df[self.item_id_field].isin(selected_items), [self.item_id_field, "rank"]
                ].values
                agg_sim_score = [self.item_matrix[i, selected_items].sum() for i in candidate_items[:, 0]]
                sim_ranks = rankdata(agg_sim_score, method="min")

                weighted_scores = theta * candidate_items[:, 1] + (1 - theta) * sim_ranks
                selected_items.append(
                    candidate_items[np.argmin(weighted_scores), 0]
                )  # selected item with minimum weighted score

            # NOTE: decoding to get the original user, item IDs
            item_ids = [self.idx2movie[x] for x in selected_items]

            results.append(
                {
                    "user": user_id,
                    "reranked_items": item_ids,
                }
            )

        # Restructure results into a DataFrame
        print("Collecting reranked results...")
        reranked_df = pd.DataFrame(results)
        print(f"Reranked DataFrame shape: {reranked_df.shape}")
        reranked_df = reranked_df.merge(
            self.eval_df[["user", "rec_items", "gt_items"]], on="user", how="inner"
        )
        reranked_df = reranked_df.rename(
            columns={"rec_items": "asis_rec_items", "reranked_items": "rec_items"}
        )
        print(f"Final Reranked DataFrame shape: {reranked_df.shape}")

        return reranked_df
