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
YEAR_FIELD = "date_year"
TIMESTAMP_FIELD = "timestamp"
RATING_FIELD = "rating"
LABEL_FIELD = "label"
FEATURE_FIELD = ["actorID", "country", "directorID", "genre"]
FEATURE_IDX_FIELD = ["actorID_idx", "country_idx", "directorID_idx", "genre_idx"]
IS_LIST_FEATURES = [True, False, False, True]

# TODO: change this to assign other dir for user/item id mappling
USER_MAPPING_DIR = "../datasets/userid_mapping.csv"
ITEM_MAPPING_DIR = "../datasets/itemid_mapping.csv"


"""
Toolkits:
- `set_seed`: Set random seed for reproducibility.
- `DataPreprocessor`: Class for data preprocessing.
    - `load_and_process_df`: Load and process the dataset.
        - `_get_illegal_ids_by_inter_num`: Get illegal ids by interaction number.
        - `_filter_by_threshold`: Filter out user/item with interactions less than min threshold.
- `FeatureEngineer`: Class for feature engineering.
    - `join_item_features`: Join item features (actor, country, director, genre) to the DataFrame.
        - `_extract_top_k_actors`: Extract the top K actors for each movie.
        - `_extract_country`: Extract the country information from the country data file.
        - `_extract_director`: Extract the director information from the director data file.
        - `_extract_genres`: Extract the genre information from the genre data file.
    - `encode_category_feature`: Encode categorical features using one-hot encoding.
- `ExperimentDataPreprocessor`: Class for experiment preparation.
    - `stratified_time_split`: Split the dataset into train, validation, and test sets based on user interactions over time.
    - `create_interaction_graph`: Create a user-item interaction graph from the DataFrame.
    - `prepare_prediction_df`: Prepare the prediction DataFrame for the test set.
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
        rating_threshold: float = RATING_THRESHOLD,
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
        # NOTE: get the timestamp / label the target based on the rating threshold
        df[TIMESTAMP_FIELD] = pd.to_datetime(
            {
                "year": df["date_year"],
                "month": df["date_month"],
                "day": df["date_day"],
                "hour": df["date_hour"],
                "minute": df["date_minute"],
                "second": df["date_second"],
            }
        )
        df[LABEL_FIELD] = (df[rating_col] >= rating_threshold).astype(int)
        # NOTE: Final Data Info
        print("==== Final Data Info: ====")
        print("Data Year Range:", year_range)
        print("Rating Threshold:", rating_threshold)
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

    def join_item_features(
        self,
        df: pd.DataFrame,
        actor_file_dir: str = ACTOR_DATA_DIR,
        country_file_dir: str = COUNTRY_DATA_DIR,
        director_file_dir: str = DIRECTOR_DATA_DIR,
        genre_file_dir: str = GENRE_DATA_DIR,
        actor_k: int = 5,
        pad_token: str = "[PAD]",
    ) -> pd.DataFrame:
        """
        Join item features (actor, country, director, genre) to the DataFrame.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            actor_file_dir (str): The path to the actor data file.
            country_file_dir (str): The path to the country data file.
            director_file_dir (str): The path to the director data file.
            genre_file_dir (str): The path to the genre data file.
            actor_k (int): The number of top actors to extract.
            pad_token (str): The padding token to use for categorical features.
        Returns:
            pd.DataFrame: The DataFrame with joined item features.
        """
        # NOTE: Extract features
        print("extracting item features...")
        actor_df = self._extract_top_k_actors(file_dir=actor_file_dir, K=actor_k, pad_token=pad_token)
        country_df = self._extract_country(file_dir=country_file_dir)
        director_df = self._extract_director(file_dir=director_file_dir)
        genre_df = self._extract_genres(file_dir=genre_file_dir, pad_token=pad_token)

        # NOTE: Merge features (inner join)
        print("merging features...")
        print("interaction data count before merging:", len(df))
        df = df.merge(actor_df, on="movieID")
        df = df.merge(country_df, on="movieID")
        df = df.merge(director_df, on="movieID")
        df = df.merge(genre_df, on="movieID")
        print("interaction data count after merging:", len(df))
        print("done!")

        return df

    def _extract_top_k_actors(
        self, file_dir: str = ACTOR_DATA_DIR, K: int = 5, pad_token: str = "[PAD]"
    ) -> pd.DataFrame:
        """
        Extract the top K actors for each movie from the actor data file. (1-to-N mapping)
        Args:
            file_dir (str): The path to the actor data file.
            K (int): The number of top actors to extract.
            pad_token (str): The padding token to use for actors.
        Returns:
            pd.DataFrame: A DataFrame containing the top K actors for each movie.
        """
        actor_df = pd.read_table(file_dir, encoding="latin-1")
        return (
            actor_df.groupby("movieID")
            .apply(
                lambda df: (
                    df.sort_values("ranking").head(K)["actorID"].tolist() + [pad_token] * max(0, K - len(df))
                )[:K]
            )  # ensure no longer than K
            .reset_index(name="actorID")
        )

    def _extract_country(self, file_dir: str = COUNTRY_DATA_DIR) -> pd.DataFrame:
        """
        Extract the country information from the country data file. (1-to-1 mapping)
        Args:
            file_dir (str): The path to the country data file.
        Returns:
            pd.DataFrame: A DataFrame containing the country information for each movie.
        """
        country_df = pd.read_table(file_dir)
        return country_df

    def _extract_director(self, file_dir: str = DIRECTOR_DATA_DIR) -> pd.DataFrame:
        """
        Extract the director information from the director data file. (1-to-1 mapping)
        Args:
            file_dir (str): The path to the director data file.
        Returns:
            pd.DataFrame: A DataFrame containing the director information for each movie.
        """
        director_df = pd.read_table(file_dir, encoding="latin-1")
        return director_df

    def _extract_genres(
        self, file_dir: str = GENRE_DATA_DIR, K: int = 8, pad_token: str = "[PAD]"
    ) -> pd.DataFrame:
        """
        Extract the genre information from the genre data file. (1-to-N mapping)
        Args:
            file_dir (str): The path to the genre data file.
            K (int): The max number of genres (by EDA).
            pad_token (str): The padding token to use for genres.
        Returns:
            pd.DataFrame: A DataFrame containing the genre information for each movie.
        """
        genre_df = pd.read_table(file_dir)

        return (
            genre_df.groupby("movieID")
            .apply(
                lambda df: (df.head(K)["genre"].tolist() + [pad_token] * max(0, K - len(df)))[
                    :K
                ]  # ensure no longer than K
            )
            .reset_index(name="genre")
        )

class FeatureEngineer:
    """
    Feature Engineer for MovieLens dataset.
    """

    def __init__(self):
        self.vocab2idx = {}
        self.pad_token = "[PAD]"
        self.oov_token = "[OOV]"

    def fit_transform(
        self,
        df: pd.DataFrame,
        user_col: str = USER_ID_FIELD,
        item_col: str = ITEM_ID_FIELD,
        feature_cols: List[str] = FEATURE_FIELD,
        is_list_features: List[bool] = IS_LIST_FEATURES,
        drop_original: bool = True,
        user_mapping_file: str = USER_MAPPING_DIR,
        item_mapping_file: str = ITEM_MAPPING_DIR,
    ) -> pd.DataFrame:
        """
        Fit and transform categorical features to create a mapping from vocab to index.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            user_col (str): The column name for user IDs.
            item_col (str): The column name for item IDs.
            feature_cols (List[str]): The column names for the categorical features to fit and transform.
            is_list_features (List[bool]): Whether the columns contain lists of categorical features.
            drop_original (bool): Whether to drop the original columns after encoding.
        Returns:
            pd.DataFrame: The DataFrame with the encoded categorical features.
        """
        self.fit(
            df,
            user_col,
            item_col,
            feature_cols,
            is_list_features,
            user_mapping_file,
            item_mapping_file,
        )
        return self.transform(
            df,
            user_col,
            item_col,
            feature_cols,
            is_list_features,
            drop_original,
        )

    def fit(
        self,
        df: pd.DataFrame,
        user_col: str = USER_ID_FIELD,
        item_col: str = ITEM_ID_FIELD,
        feature_cols: List[str] = FEATURE_FIELD,
        is_list_features: List[bool] = IS_LIST_FEATURES,
        user_mapping_file: str = USER_MAPPING_DIR,
        item_mapping_file: str = ITEM_MAPPING_DIR,
    ):
        """
        Fit categorical features.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            user_col (str): The column name for user IDs.
            item_col (str): The column name for item IDs.
            feature_cols (List[str]): The column names for the categorical features to fit.
            is_list_features (bool): Whether the columns contain lists of categorical features.
        Returns:
            Dict[str, Dict[str, int]]: A dictionary containing the vocab2idx mapping for each column.
        """
        # NOTE: fit user_id and item_id
        self.vocab2idx[user_col], self.vocab2idx[item_col] = self._fit_re_index(
            df, user_col, item_col, user_mapping_file, item_mapping_file
        )
        print("Fitted: user/item mapping")

        # NOTE: fit categorical features
        for col, is_list in zip(feature_cols, is_list_features):
            self.vocab2idx[col] = self._fit_category_feature(df, col, is_list)
            print("Fitted: vocab2idx for", col)

    def _fit_re_index(
        self,
        df: pd.DataFrame,
        user_col: str = USER_ID_FIELD,
        item_col: str = ITEM_ID_FIELD,
        user_mapping_file: str = USER_MAPPING_DIR,
        item_mapping_file: str = ITEM_MAPPING_DIR,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Re-index the user and item mapping after joining and filtering.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            user_col (str): The column name for user IDs.
            item_col (str): The column name for item IDs.
        Returns:
            Tuple[Dict[str, int], Dict[str, int]]: A tuple containing the user and item mapping dictionaries.
        """
        # NOTE: Re-index
        uni_users = sorted(df[user_col].unique().tolist())
        uni_items = sorted(df[item_col].unique().tolist())
        # start from 0
        user_vocab2idx = {k: i for i, k in enumerate(uni_users)}
        item_vocab2idx = {k: i for i, k in enumerate(uni_items)}
        # add oov token
        user_vocab2idx[self.oov_token] = len(user_vocab2idx)
        item_vocab2idx[self.oov_token] = len(item_vocab2idx)

        # dump to files
        u_df = pd.DataFrame(list(user_vocab2idx.items()), columns=["from", "to"])
        i_df = pd.DataFrame(list(item_vocab2idx.items()), columns=["from", "to"])
        u_df.to_csv(os.path.join(user_mapping_file), index=False)
        i_df.to_csv(os.path.join(item_mapping_file), index=False)
        print(f"Re-index mapping dumped into ...")
        print(f"user: {user_mapping_file}")
        print(f"item: {item_mapping_file}")
        
        return user_vocab2idx, item_vocab2idx

    def _fit_category_feature(
        self,
        df: pd.DataFrame,
        col: str,
        is_list: bool,
    ):
        """
        Fit categorical features to create a mapping from vocab to index.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            col (str): The column name for the categorical feature to fit.
            is_list (bool): Whether the column contains lists of categorical features.
        Returns:
            Dict[str, int]: A dictionary containing the vocab2idx mapping for the column.
        """
        if is_list:
            # NOTE: Flatten all actor/genre lists and build unique vocabulary
            vocabs = set(str(cat_feat) for cat_list in df[col] for cat_feat in cat_list)
            vocabs -= set([self.pad_token])  # remove padding token
        else:
            vocabs = set(df[col].astype(str))

        # NOTE: Create a mapping from vocab to index
        vocab2idx = {vocab: idx + 1 for idx, vocab in enumerate(sorted(vocabs))}
        vocab2idx[self.pad_token] = 0  # for paddings in train/test set
        vocab2idx[self.oov_token] = len(vocab2idx) + 1  # for unknowns features in test set
        return vocab2idx

    def transform(
        self,
        df: pd.DataFrame,
        user_col: str = USER_ID_FIELD,
        item_col: str = ITEM_ID_FIELD,
        feature_cols: List[str] = FEATURE_FIELD,
        is_list_features: List[bool] = IS_LIST_FEATURES,
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """
        Encode categorical features.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            user_col (str): The column name for user IDs.
            item_col (str): The column name for item IDs.
            feature_cols (List[str]): The column names for the categorical features to encode.
            is_list_features (List[bool]): Whether the columns contain lists of categorical features.
            drop_original (bool): Whether to drop the original columns after encoding.
        Returns:
            pd.DataFrame: The DataFrame with the encoded categorical features.
        """
        df = self._transfrom_re_index(df, user_col, item_col)
        print("Transformed: Re-index user/item mapping")

        for col, is_list in zip(feature_cols, is_list_features):
            df = self._encode_category_feature(df, col, is_list, drop_original)
            print("Transformed: Encoded idx for", col)
        return df

    def _transfrom_re_index(
        self,
        df: pd.DataFrame,
        user_col: str = USER_ID_FIELD,
        item_col: str = ITEM_ID_FIELD,
    ) -> pd.DataFrame:
        """
        Re-index the user and item mapping after joining and filtering.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            user_col (str): The column name for user IDs.
            item_col (str): The column name for item IDs.
        Returns:
            pd.DataFrame: The DataFrame with re-indexed user and item mapping.
        """

        if user_col not in self.vocab2idx or item_col not in self.vocab2idx:
            raise ValueError(f"Column {user_col} or {item_col} not fitted. Please fit the column first.")

        user_vocab2idx: dict = self.vocab2idx[user_col]
        item_vocab2idx: dict = self.vocab2idx[item_col]

        df[user_col] = df[user_col].apply(lambda x: user_vocab2idx.get(x, user_vocab2idx[self.oov_token]))
        df[item_col] = df[item_col].apply(lambda x: item_vocab2idx.get(x, item_vocab2idx[self.oov_token]))
        df[user_col] = df[user_col].astype(int)
        df[item_col] = df[item_col].astype(int)
        return df

    def _encode_category_feature(
        self, df: pd.DataFrame, col: str, is_list: bool, drop_original: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features using fitted vocab2idx.
        Args:
            df (pd.DataFrame): The input DataFrame containing user-item interactions.
            col (str): The column name for the categorical feature to encode.
            is_list (bool): Whether the column contains lists of categorical features.
            drop_original (bool): Whether to drop the original column after encoding.
        Returns:
            pd.DataFrame: The DataFrame with the encoded categorical feature.
        """
        # NOTE: Check if the column already fitted
        if col not in self.vocab2idx:
            raise ValueError(f"Column {col} not fitted. Please fit the column first.")
        vocab2idx = self.vocab2idx[col]

        # NOTE: Encode the categorical feature
        if is_list:
            df[f"{col}_idx"] = df[col].apply(
                lambda x: [
                    vocab2idx[cat_feat] if cat_feat in vocab2idx else vocab2idx[self.oov_token]
                    for cat_feat in x
                ]
            )
        else:
            df[f"{col}_idx"] = df[col].apply(
                lambda x: vocab2idx[x] if x in vocab2idx else vocab2idx[self.oov_token]
            )

        # NOTE: Drop the original column if specified
        if drop_original:
            df = df.drop(columns=[col])

        return df


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
        print("Building bi-directed edges (duplications)...")
        edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)  # [2, num_edges*2]
        print("Building labels...")
        edge_label = torch.tensor(df_split[LABEL_FIELD].values, dtype=torch.float)
        print("Building bi-directed labels (duplications)...", "\n")
        edge_label = torch.cat((edge_label, edge_label))
        data = Data(edge_index=edge_index, edge_label=edge_label)
        print("Interaction Graph:", data)
        print("Edge Index:", data.edge_index)

        return data

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
