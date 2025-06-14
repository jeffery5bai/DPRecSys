# cache_pipeline.py
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import joblib
import numpy as np
import pandas as pd
import torch
from common.eval import Evaluator
from common.exp_data_utils import ExperimentDataPreprocessor
from common.utils import DataPreprocessor, FeatureEngineer
from tqdm import tqdm

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

RANDOM_SEED = 42


def save_np_array(data, path):
    np.save(path, data)
    print(f"Saved numpy array to {path}")


def run_data_preprocessing(data_path):
    data_preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    experiment_data_preprocessor = ExperimentDataPreprocessor()
    evaluator = Evaluator()

    # Load and join metadata
    interaction_df = data_preprocessor.load_and_process_df(file_dir=data_path, year_range=(2006, 2008))
    interaction_info_df = data_preprocessor.join_item_features(df=interaction_df, actor_k=5, threshold=5)

    # Time split
    train_df, valid_df, test_df = experiment_data_preprocessor.stratified_time_split(
        interaction_info_df,
        time_col="timestamp",
        train_ratio=0.75,
        val_ratio=0.1,
        test_ratio=0.15,
    )

    # Encode
    encoded_train_df = feature_engineer.fit_transform(train_df)
    encoded_valid_df = feature_engineer.transform(valid_df)
    encoded_test_df = feature_engineer.transform(test_df)

    # Save vocab
    joblib.dump(feature_engineer.vocab2idx, f"{CACHE_DIR}/vocab2idx.pkl")

    # Save encoded dfs
    encoded_train_df.to_parquet(f"{CACHE_DIR}/encoded_train_df.parquet")
    encoded_valid_df.to_parquet(f"{CACHE_DIR}/encoded_valid_df.parquet")
    encoded_test_df.to_parquet(f"{CACHE_DIR}/encoded_test_df.parquet")

    # Graph
    vocab = feature_engineer.vocab2idx
    train_graph = experiment_data_preprocessor.create_knowledge_graph(encoded_train_df, vocab)
    torch.save(train_graph, f"{CACHE_DIR}/train_hetero_graph.pt")

    # DPS, item multihot vectors
    user_dps_df = evaluator.eval_user_diversity_preference_scale(encoded_train_df, vocab, normalized=True)
    item_vec_df = evaluator.get_item_feature_multihot_vec(encoded_train_df, vocab)

    # Save DPS and item vec dfs (full)
    user_dps_df.to_parquet(f"{CACHE_DIR}/user_dps_df.parquet")
    item_vec_df.to_parquet(f"{CACHE_DIR}/item_vec_df.parquet")

    # ----------------------------------------------------
    # 新增邏輯：分別把 actor_ids, director_ids, country_ids, genre_ids（以及 dps, vec, wvec）獨立抽出並存成 numpy 檔案 (train/valid)
    # ----------------------------------------------------

    # 欄位名稱
    ENCODED_FIELDS = {
        "actor": "actorID_idx",
        "country": "country_idx",
        "director": "directorID_idx",
        "genre": "genre_idx",
    }

    DPS_FIELDS = {
        "actor": "actorID_dps",
        "country": "country_dps",
        "director": "directorID_dps",
        "genre": "genre_dps",
    }

    VEC_FIELDS = ["actor_vec", "country_vec", "director_vec", "genre_vec"]
    WVEC_FIELDS = ["actor_wvec", "country_wvec", "director_wvec", "genre_wvec"]

    # 定義函式：將 dataframe 某欄是 list/np.array 的欄位，轉成 np.array
    def df_col_to_np_array(df, col_name):
        # 對於每一列是 list/np.array 的欄位，統一轉成 np.array 2D 或 3D (視情況)
        # 例如 vec 和 wvec 是多維向量，先用 np.stack
        # 這邊直接用 np.stack，因為 vec, wvec 是 list of arrays
        return np.stack(df[col_name].values)

    train_triplet_df = experiment_data_preprocessor.prepare_triplet_df(encoded_train_df, k_negative_samples=5)
    train_triplet_with_dps_df = train_triplet_df.merge(user_dps_df, on="userID", how="left")
    train_triplet_with_dps_df = train_triplet_with_dps_df.merge(
        item_vec_df, left_on="pos_item_id", right_on="movieID", how="left"
    )
    train_triplet_with_dps_df.to_parquet(f"{CACHE_DIR}/train_triplet_with_dps.parquet")

    valid_triplet_df = experiment_data_preprocessor.prepare_triplet_df(encoded_valid_df, k_negative_samples=5)
    valid_triplet_with_dps_df = valid_triplet_df.merge(user_dps_df, on="userID", how="inner")
    valid_triplet_with_dps_df = valid_triplet_with_dps_df.merge(
        item_vec_df, left_on="pos_item_id", right_on="movieID", how="inner"
    )
    valid_triplet_with_dps_df.to_parquet(f"{CACHE_DIR}/valid_triplet_with_dps.parquet")

    prediction_pool_df = experiment_data_preprocessor.prepare_prediction_df(encoded_test_df, K=500)
    prediction_pool_df.to_parquet(f"{CACHE_DIR}/prediction_pool_df.parquet")

    for split_name, df in zip(
        ["train", "valid", "prediction"],
        [train_triplet_with_dps_df, valid_triplet_with_dps_df, prediction_pool_df],
    ):

        # 1. 儲存 actor_ids, country_ids, director_ids, genre_ids
        for feat, col in ENCODED_FIELDS.items():
            if col in df.columns:
                # 針對欄位是 list or np.array 的特殊處理
                arr = df_col_to_np_array(df, col)
                save_np_array(arr, f"{CACHE_DIR}/{split_name}_{feat}_ids.npy")

        # 2. 儲存 DPS
        if split_name in ["train", "valid"]:
            for feat, col in DPS_FIELDS.items():
                if col in df.columns:
                    save_np_array(df[col].values, f"{CACHE_DIR}/{split_name}_{feat}_dps.npy")

            # 3. 儲存 vec, wvec (item 多熱向量特徵)
            for vec_col in VEC_FIELDS + WVEC_FIELDS:
                if vec_col in df.columns:
                    arr = df_col_to_np_array(df, vec_col)
                    save_np_array(arr, f"{CACHE_DIR}/{split_name}_{vec_col}.npy")

    print("✅ Data cached successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="../datasets/hetrec2011-movielens-2k-v2/user_ratedmovies.dat"
    )
    args = parser.parse_args()
    run_data_preprocessing(args.data_path)
