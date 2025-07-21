"""
DPA-RS (Diversity Preference-Aware Recommender System) Reranker.
Yin, K., Fang, X., Chen, B., & Sheng, O. R. L. (2023). Diversity preference-aware link recommendation for online social networks. 
Information Systems Research, 34(4), 1398-1414.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from typing import Any, Dict, List

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm

from post_processing.abstract_reranker import Reranker

class DPA_RS(Reranker):
    """
    DPA-RS (Differentially Private Attribute-based Recommender System) Reranker.
    This class implements the DPA-RS algorithm for reranking items based on user preferences and item attributes.
    """

    def __init__(self, ground_truth_dps_df, **kwargs):
        """
        Initialize the DPA_RS reranker with configuration parameters.

        Args:
            config (dict): Configuration parameters for the reranker.
        """
        super().__init__(**kwargs)
        self.ground_truth_dps_df: pd.DataFrame = ground_truth_dps_df

    def dpars_iterative_solver(
        self,
        C_list: List[np.ndarray],
        d_list: List[np.ndarray],
        k: int,
        epsilon: float = 1e-3,
        max_iter: int = 100,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Full DPA-RS iterative solver for a single user.

        Args:
            C_list (List[np.ndarray]): List of item attribute matrices (C_h), shape (attr_dim_h, m)
            d_list (List[np.ndarray]): List of user preference vectors (d_h), shape (attr_dim_h,)
            k (int): Number of items to select
            epsilon (float): Convergence threshold for delta
            max_iter (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility

        Returns:
            y_final (np.ndarray): Final relaxed selection vector (in [0,1])
        """
        H = len(C_list)

        # NOTE: Random initialization of beta and gamma
        np.random.seed(random_state)  # For reproducibility
        beta = np.random.rand(H)
        gamma = np.random.rand(H)

        for _ in range(max_iter):
            # NOTE: solve problem (4)
            y = self._solve_dpa_lagrangian(C_list, d_list, beta, gamma, k)

            # NOTE: compute delta vector (size 2H)
            delta = []
            for h in range(H):
                C_h = C_list[h]
                d_h = d_list[h]
                d_h_bar = d_h / np.linalg.norm(d_h)
                Cy = C_h @ y
                norm_Cy = np.linalg.norm(Cy)

                delta_h = beta[h] * norm_Cy - d_h_bar @ Cy  # Eq (7)
                delta_H_plus_h = gamma[h] * norm_Cy - 1  # Eq (8)
                delta.extend([delta_h, delta_H_plus_h])

            delta = np.array(delta)

            # NOTE: Check convergence
            if np.linalg.norm(delta) < epsilon:
                break

            # NOTE: update beta and gamma using Eq (9, 10)
            for h in range(H):
                C_h = C_list[h]
                d_h = d_list[h]
                d_h_bar = d_h / np.linalg.norm(d_h)
                Cy = C_h @ y
                norm_Cy = np.linalg.norm(Cy)

                if norm_Cy > 1e-8:  # Avoid divide-by-zero by keeping values unchanged
                    beta[h] = d_h_bar @ Cy / norm_Cy
                    gamma[h] = 1.0 / norm_Cy

        return y

    def _solve_dpa_lagrangian(self, C_list, d_list, beta, gamma, k) -> np.ndarray:
        """
        Solve one iteration of the DPA-RS Lagrangian optimization problem (problem (4)).

        Args:
            C_list (List[np.ndarray]): List of item attribute matrices (C_h), shape (attr_dim_h, m)
            d_list (List[np.ndarray]): List of user preference vectors (d_h), shape (attr_dim_h,)
            beta (np.ndarray): Current beta vector, shape (H,)
            gamma (np.ndarray): Current gamma vector, shape (H,)
            k (int): Number of items to select

        Returns:
            y_opt (np.ndarray): Optimal relaxed selection vector, shape (m,)
        """
        RESIDUAL = 1e-4 # to avoid division by zero
        H = len(C_list)
        m = C_list[0].shape[1]  # number of candidate items

        # NOTE: Define the optimization variable
        y = cp.Variable(m)

        objective_terms = []
        for h in range(H):
            d_h_bar = cp.Constant(d_list[h] / max(np.linalg.norm(d_list[h]), RESIDUAL))  
            C_h = cp.Constant(C_list[h])
            beta_h = cp.Constant(beta[h])
            gamma_h = cp.Constant(gamma[h])

            Cy = C_h @ y
            norm_Cy = cp.maximum(cp.norm(Cy, 2), RESIDUAL)
            dot_product = cp.matmul(d_h_bar, Cy)

            term = gamma_h * (dot_product - beta_h * norm_Cy)
            objective_terms.append(term)

        # Add small L2 regularization to avoid flatness
        regularizer = RESIDUAL * cp.norm(y, 2)

        # NOTE: Objective function and constraints
        objective = cp.Maximize(cp.sum(objective_terms) - regularizer) # Eq (4)
        constraints = [cp.sum(y) == k, y >= 0, y <= 1]

        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
        except cp.SolverError:
            problem.solve(solver=cp.SCS)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"CVXPY solver failed: {problem.status}")

        return y.value
    
    def prepare_input(self) -> pd.DataFrame:
        encoded_df = self.preprocess()

        item_feature_vec_df = self.evaluator.get_item_feature_multihot_vec(
            df_with_encoded_features=encoded_df,
            feature_vocab2idx=self.feature_engineer.vocab2idx,
        )
        item_feature_vec_df = item_feature_vec_df[[self.item_id_field] + [f"{feat}_vec"  for feat in self.feature_fields]]

        # NOTE: merge the attribute vectors
        combined_df = encoded_df.merge(
            item_feature_vec_df,
            on=self.item_id_field,
            how="inner"
        )

        gt_dps_df = self.ground_truth_dps_df[[self.user_id_field] + [f"{feat}_wvec" for feat in self.feature_fields]].copy()
        # NOTE: merge the user preference vectors
        combined_df = combined_df.merge(
            gt_dps_df,
            on=self.user_id_field,
            how="inner"
        )

        combined_df = combined_df.rename(
            columns={f"{feat}_wvec": f"{feat}_gt" for feat in self.feature_fields},
        )
        print(f"Combined DataFrame shape: {combined_df.shape}")

        return combined_df


    def rerank(self, top_k: int = 20, max_iter: int = 100, random_state: int = 42) -> pd.DataFrame:

        """
        Apply DPA-RS re-ranking to each user's candidate items.

        Args:
            top_k (int): Number of items to recommend
            max_iter (int): Maximum number of iterations for the DPA-RS solver
            random_state (int): Random seed for reproducibility

        Returns:
            pd.DataFrame: New re-ranked recommendations per user
        """

        # NOTE: this dataframe has been encoded and contains all necessary features
        print("Preparing input DataFrame for DPA-RS...")
        df = self.prepare_input()
        
        results = []
        print("Reranking items for each user...")
        for user_id, user_df in tqdm(df.groupby("userID"), desc="Reranking users"):
            # Build C matrix: shape (feat_dim, num_items)
            C = [
                np.stack(user_df[f"{feat}_vec"].to_numpy()).T  # shape (feat_dim, num_items)
                for feat in self.feature_fields
            ]

            # User preference vector
            d = [
                user_df[f"{feat}_gt"].values[0]  # shape (feat_dim,)
                for feat in self.feature_fields
            ]

            # Solve relaxed DPA-RS
            y = self.dpars_iterative_solver(C_list=C, d_list=d, k=top_k, max_iter=max_iter, random_state=random_state)

            # Top-k item indices in descending score
            top_k_indices = np.argsort(-y)[:top_k]

            # Map back to item IDs
            item_ids = user_df.iloc[top_k_indices][self.item_id_field].tolist()

            # NOTE: decoding to get the original user, item IDs
            user_id = self.feature_engineer.idx2vocab[self.user_id_field][user_id]
            item_ids = [
                self.feature_engineer.idx2vocab[self.item_id_field][x]
                for x in item_ids
            ]

            results.append({
                "user": user_id,
                "reranked_items": item_ids
            })

        # Restructure results into a DataFrame
        print("Collecting reranked results...")
        reranked_df = pd.DataFrame(results)
        print(f"Reranked DataFrame shape: {reranked_df.shape}")
        reranked_df = reranked_df.merge(
            self.eval_df[["user", "rec_items", "gt_items"]],
            on="user",
            how="inner"
        )
        reranked_df = reranked_df.rename(columns={"rec_items": "asis_rec_items", "reranked_items": "rec_items"})
        print(f"Final Reranked DataFrame shape: {reranked_df.shape}")

        return reranked_df