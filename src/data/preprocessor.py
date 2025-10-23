import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MovieLensPreprocessor:
    """Preprocessor for MovieLens dataset"""

    # Khởi tạo các components
    def __init__(self):
        # Sklearn class để encode categorical → numerical
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.rating_scaler = None
        self.is_fitted = False

        # Statistics
        self.n_users = 0
        self.n_items = 0
        self.rating_stats = {}

    def fit(self, ratings_df: pd.DataFrame) -> None:
        """Fit preprocessor on ratings data --- data leakage"""
        logger.info("Fitting preprocessor trên ratings data...")

        # Fit encoders
        # Lấy unique users/items
        unique_users = ratings_df['user_id'].unique()
        unique_items = ratings_df['movie_id'].unique()

        # Create continuous indices starting from 0
        # Fit LabelEncoder
        self.user_encoder.fit(unique_users)
        self.item_encoder.fit(unique_items)

        # Store statistics
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)

        # Rating statistics
        self.rating_stats = {
            'min': ratings_df['rating'].min(),
            'max': ratings_df['rating'].max(),
            'mean': ratings_df['rating'].mean(),
            'std': ratings_df['rating'].std()
        }

        self.is_fitted = True
        logger.info(f"Preprocessor fitted: {self.n_users} users, {self.n_items} items")

    def transform_ratings(self, ratings_df: pd.DataFrame,
                          normalize_ratings: bool = False) -> np.ndarray:
        """Transform ratings DataFrame to numpy array with encoded IDs"""
        # Check is_fitted?
        if not self.is_fitted:
            raise ValueError("Preprocessor phải được fitting trước khi transform")

        # Tạo copy tránh sửa đổi bản gốc
        df = ratings_df.copy()

        # Transform user and item IDs
        # Apply user_encoder.transform()
        df['user_id_encoded'] = self.user_encoder.transform(df['user_id'])
        # Apply item_encoder.transform()
        df['movie_id_encoded'] = self.item_encoder.transform(df['movie_id'])

        # Optionally normalize ratings
        ratings = df['rating'].values
        if normalize_ratings:
            if self.rating_scaler is None:
                # StandardScaler
                self.rating_scaler = StandardScaler()
                ratings = self.rating_scaler.fit_transform(ratings.reshape(-1, 1)).flatten()
            else:
                ratings = self.rating_scaler.transform(ratings.reshape(-1, 1)).flatten()

        # Return as numpy array: [user_id, item_id, rating]
        result = np.column_stack([
            df['user_id_encoded'].values,
            df['movie_id_encoded'].values,
            ratings
        ])

        return result
        # Neural networks --- normalize

    def fit_transform(self, ratings_df: pd.DataFrame,
                      normalize_ratings: bool = False) -> np.ndarray:
        """Fit + Transform data """
        self.fit(ratings_df)
        return self.transform_ratings(ratings_df, normalize_ratings)

    def inverse_transform_ratings(self, ratings_array: np.ndarray) -> np.ndarray:
        """Chuyển ratings đã normalize về scale gốc --- retrun gốc"""
        if self.rating_scaler is not None:
            ratings = self.rating_scaler.inverse_transform(
                ratings_array[:, 2].reshape(-1, 1)
            ).flatten()
            result = ratings_array.copy()
            result[:, 2] = ratings
            return result
        return ratings_array

    def create_train_test_split(self, ratings_df: pd.DataFrame,
                                test_size: float = 0.2,
                                random_state: int = 42,
                                stratify_by_user: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chia train/test"""
        logger.info(f"Tạo train/test split (test_size={test_size})")

        if stratify_by_user:
            # Đảm bảo mỗi user có ratings trong cả train và test
            train_dfs = []
            test_dfs = []

            # Stratified by User
            for user_id in ratings_df['user_id'].unique():
                user_ratings = ratings_df[ratings_df['user_id'] == user_id]

                if len(user_ratings) < 2:
                    # Nếu user chỉ có 1 rating, cho vào train
                    train_dfs.append(user_ratings)
                else:
                    user_train, user_test = train_test_split(
                        user_ratings, test_size=test_size, random_state=random_state
                    )
                    train_dfs.append(user_train)
                    test_dfs.append(user_test)

            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
        else:
            # Simple Random Split
            train_df, test_df = train_test_split(
                ratings_df, test_size=test_size, random_state=random_state
            )

        logger.info(f"Split completed: {len(train_df)} train, {len(test_df)} test samples")
        return train_df, test_df

    def create_negative_samples(self, ratings_df: pd.DataFrame,
                                negative_ratio: float = 1.0,
                                random_state: int = 42) -> pd.DataFrame:
        """Tạo mẫu âm cho implicit feedback learning"""
        logger.info(f"Creating negative samples (ratio={negative_ratio})")

        np.random.seed(random_state)

        # Lấy tất cả positive interactions
        # check membership
        positive_interactions = set(
            zip(ratings_df['user_id'], ratings_df['movie_id'])
        )

        # Lấy tất cả users và items
        all_users = ratings_df['user_id'].unique()
        all_items = ratings_df['movie_id'].unique()

        # Sinh negative samples
        n_negatives = int(len(ratings_df) * negative_ratio)
        negative_samples = []

        while len(negative_samples) < n_negatives:
            user = np.random.choice(all_users)
            item = np.random.choice(all_items)

            if (user, item) not in positive_interactions:
                # chưa tương tác
                negative_samples.append({
                    'user_id': user,
                    'movie_id': item,
                    'rating': 0,  # Implicit negative
                    'timestamp': pd.NaT
                })

        negative_df = pd.DataFrame(negative_samples)

        # Kết hợp với positive samples
        ratings_with_negatives = pd.concat([ratings_df, negative_df], ignore_index=True)

        logger.info(f"Added {len(negative_samples)} negative samples")
        return ratings_with_negatives

    def get_user_item_matrix(self, ratings_array: np.ndarray,
                             fill_value: float = 0.0) -> np.ndarray:
        """Tạo ma trận User-Item từ array đã encode"""
        matrix = np.full((self.n_users, self.n_items), fill_value, dtype=np.float32)

        for user_id, item_id, rating in ratings_array:
            matrix[int(user_id), int(item_id)] = rating

        return matrix

    def filter_data(self, ratings_df: pd.DataFrame,
                    min_user_ratings: int = 5,
                    min_item_ratings: int = 5) -> pd.DataFrame:
        """Lọc bỏ users/items có quá ít ratings"""
        logger.info(f"Filtering data (min_user_ratings={min_user_ratings}, "
                    f"min_item_ratings={min_item_ratings})")

        original_size = len(ratings_df)

        # Lọc lặp lại tới hội tụ
        prev_size = 0
        current_df = ratings_df.copy()

        while len(current_df) != prev_size:
            prev_size = len(current_df)

            # Lọc users có quá ít ratings
            user_counts = current_df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_user_ratings].index
            current_df = current_df[current_df['user_id'].isin(valid_users)]

            # Lọc items có quá ít ratings
            item_counts = current_df['movie_id'].value_counts()
            valid_items = item_counts[item_counts >= min_item_ratings].index
            current_df = current_df[current_df['movie_id'].isin(valid_items)]

        logger.info(f"Dữ liệu sau khi lọc: {original_size} -> {len(current_df)} ratings")
        logger.info(f"Users: {ratings_df['user_id'].nunique()} -> {current_df['user_id'].nunique()}")
        logger.info(f"Items: {ratings_df['movie_id'].nunique()} -> {current_df['movie_id'].nunique()}")

        return current_df

    def get_statistics(self, ratings_df: pd.DataFrame) -> Dict[str, Any]:
        """Tính toán comprehensive statistics"""
        stats = {
            # Basic counts
            'n_ratings': len(ratings_df),
            'n_users': ratings_df['user_id'].nunique(),
            'n_items': ratings_df['movie_id'].nunique(),
            # Rating distribution
            'rating_stats': {
                'min': ratings_df['rating'].min(),
                'max': ratings_df['rating'].max(),
                'mean': ratings_df['rating'].mean(),
                'std': ratings_df['rating'].std(),
                'distribution': ratings_df['rating'].value_counts().to_dict()
            },
            # Sparsity
            'sparsity': 1 - (len(ratings_df) / (
                    ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique()
            )),
            # User activity
            'user_activity': {
                'min': ratings_df.groupby('user_id').size().min(),
                'max': ratings_df.groupby('user_id').size().max(),
                'mean': ratings_df.groupby('user_id').size().mean(),
                'std': ratings_df.groupby('user_id').size().std()
            },
            #  Item popularity
            'item_popularity': {
                'min': ratings_df.groupby('movie_id').size().min(),
                'max': ratings_df.groupby('movie_id').size().max(),
                'mean': ratings_df.groupby('movie_id').size().mean(),
                'std': ratings_df.groupby('movie_id').size().std()
            }
        }

        return stats

    def save_preprocessor(self, filepath: str) -> None:
        """Save preprocessor state"""
        import joblib # serialize Python objects

        state = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'rating_scaler': self.rating_scaler,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'rating_stats': self.rating_stats,
            'is_fitted': self.is_fitted
        }

        joblib.dump(state, filepath)
        logger.info(f"Preprocessor saved to: {filepath}")

    @classmethod
    def load_preprocessor(cls, filepath: str) -> 'MovieLensPreprocessor':
        """Load preprocessor state"""
        import joblib

        state = joblib.load(filepath)

        preprocessor = cls()
        preprocessor.user_encoder = state['user_encoder']
        preprocessor.item_encoder = state['item_encoder']
        preprocessor.rating_scaler = state['rating_scaler']
        preprocessor.n_users = state['n_users']
        preprocessor.n_items = state['n_items']
        preprocessor.rating_stats = state['rating_stats']
        preprocessor.is_fitted = state['is_fitted']

        logger.info(f"Preprocessor loaded from: {filepath}")
        return preprocessor