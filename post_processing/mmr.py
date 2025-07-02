"""
MMR (Maximal Marginal Relevance) Reranker.
Ziegler, Cai-Nicolas, et al. "Improving recommendation lists through topic diversification."
Proceedings of the 14th international conference on World Wide Web. 2005.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import numpy as np
import pandas as pd
from post_processing.abstract_reranker import Reranker
from scipy.stats import rankdata
from tqdm import tqdm


class MMR(Reranker):
    """
    MMR (Maximal Marginal Relevance) Reranker.
    This class implements the MMR algorithm for reranking items based on diversity of item attributes.
    """

    def __init__(self, **kwargs):
        """ """
        super().__init__(**kwargs)
        # load files
        data = np.load("../experiments/artifacts/item_sim_data.npz", allow_pickle=True)
        self.item_matrix: np.array = data["sim_matrix"]
        self.movie2idx: dict = data["movie2idx"].item()
        self.idx2movie: dict = {v: k for k, v in self.movie2idx.items()}  # reverse mapping

    def rerank(self, top_k: int = 20, theta: float = 0.5) -> pd.DataFrame:
        """
        Rerank items for each user based on MMR algorithm.
        Args:
            top_k (int): Number of items to select for each user.
            theta (float): Trade-off parameter between relevance and diversity.
        Returns:
            pd.DataFrame: DataFrame containing reranked items for each user.
        """

        # NOTE: this dataframe contains all necessary features
        print("Preparing input DataFrame for MMR...")
        df = self.preprocess(encoded=False)
        # NOTE: map raw movieID to valid indices
        df[self.item_id_field] = df[self.item_id_field].map(self.movie2idx)

        results = []
        print("Reranking items for each user...")
        for user_id, user_df in tqdm(df.groupby(self.user_id_field), desc="Reranking users"):
            # initialize the selected item
            selected_items = [user_df.loc[user_df["rank"] == 1, self.item_id_field].values[0]]
            for _ in range(1, top_k):
                candidate_items = user_df.loc[
                    ~user_df[self.item_id_field].isin(selected_items), [self.item_id_field, "rank"]
                ].values
                agg_sim_score = [self.item_matrix[i, selected_items].sum() for i in candidate_items[:, 0]]
                sim_ranks = rankdata(agg_sim_score, method="min")

                weighted_scores = candidate_items[:, 1] + theta * sim_ranks
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
