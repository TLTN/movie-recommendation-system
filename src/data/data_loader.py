import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from ..utils.file_utils import ensure_dir, save_pickle, load_pickle

logger = logging.getLogger(__name__)


class MovieLensDataLoader:

    """Khởi tạo object"""
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.ratings_train = None
        self.ratings_test = None
        self.movies = None
        self.users = None
        self._user_mapping = None
        self._movie_mapping = None

    """Load dữ liệu từ file gốc"""
    def load_ratings(self, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cache_path = self.data_path.parent / "processed" / "ratings_cache.pkl"

        if use_cache and cache_path.exists():
            logger.info("Tải rating từ cache...")
            cached_data = load_pickle(cache_path)
            self.ratings_train = cached_data['train']
            self.ratings_test = cached_data['test']
            self._user_mapping = cached_data.get('user_mapping')
            self._movie_mapping = cached_data.get('movie_mapping')
        else:
            logger.info("Tải dữ liệu từ file")

            r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']

            train_path = self.data_path / "ua.base"
            test_path = self.data_path / "ua.test"

            self.ratings_train = pd.read_csv(
                train_path, sep='\t', names=r_cols, encoding='latin-1'
            )
            self.ratings_test = pd.read_csv(
                test_path, sep='\t', names=r_cols, encoding='latin-1'
            )

            # Tạo mapping
            # Gộp train + test, lấy unique + sort
            all_users = np.unique(np.concatenate([
                self.ratings_train['user_id'].values,
                self.ratings_test['user_id'].values
            ]))
            all_movies = np.unique(np.concatenate([
                self.ratings_train['movie_id'].values,
                self.ratings_test['movie_id'].values
            ]))
            # Tạo mapping dictionary
            self._user_mapping = {user_id: idx for idx, user_id in enumerate(sorted(all_users))}
            self._movie_mapping = {movie_id: idx for idx, movie_id in enumerate(sorted(all_movies))}

            # Map từ chỉ số 0
            # Aplly
            self.ratings_train['user_id'] = self.ratings_train['user_id'].map(self._user_mapping)
            self.ratings_train['movie_id'] = self.ratings_train['movie_id'].map(self._movie_mapping)
            self.ratings_test['user_id'] = self.ratings_test['user_id'].map(self._user_mapping)
            self.ratings_test['movie_id'] = self.ratings_test['movie_id'].map(self._movie_mapping)

            # Chuyển đổi timestamp sang datetime
            self.ratings_train['timestamp'] = pd.to_datetime(self.ratings_train['timestamp'], unit='s')
            self.ratings_test['timestamp'] = pd.to_datetime(self.ratings_test['timestamp'], unit='s')

            # Lưu cache
            if use_cache:
                ensure_dir(cache_path.parent)
                save_pickle({
                    'train': self.ratings_train,
                    'test': self.ratings_test,
                    'user_mapping': self._user_mapping,
                    'movie_mapping': self._movie_mapping
                }, cache_path)

        logger.info(f"Đã tải {len(self.ratings_train)} training and {len(self.ratings_test)} test ratings")

        return self.ratings_train, self.ratings_test

    def load_movies(self, use_cache: bool = True) -> pd.DataFrame:
        """Load movie metadata"""
        cache_path = self.data_path.parent / "processed" / "movies_cache.pkl"

        if use_cache and cache_path.exists():
            logger.info("Tải dữ liệu movies từ cache...")
            self.movies = load_pickle(cache_path)
        else:
            logger.info("Tải dữ liệu movie từ files...")

            # Define columns (5-19)
            item_cols = [
                'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
            ]

            movies_path = self.data_path / "u.item"
            self.movies = pd.read_csv(
                movies_path, sep='|', names=item_cols, encoding='latin-1'
            )

            # Ánh xạ movie IDs bằng mapping từ ratings
            # Apply movie mapping
            if self._movie_mapping is not None:
                self.movies['original_id'] = self.movies['movie_id']
                self.movies['movie_id'] = self.movies['movie_id'].map(self._movie_mapping)
                # Xóa các movie không có trong mapping
                self.movies = self.movies.dropna(subset=['movie_id'])
                self.movies['movie_id'] = self.movies['movie_id'].astype(int)

            # Xử lý release date
            self.movies['release_date'] = pd.to_datetime(
                self.movies['release_date'],
                format='%d-%b-%Y',
                errors='coerce'
            )
            self.movies['release_year'] = self.movies['release_date'].dt.year

            # Trích xuất main genre
            genre_cols = item_cols[5:]
            self.movies['genres'] = self.movies[genre_cols].apply(
                lambda x: [col for col, val in x.items() if val == 1], axis=1
            ) #Lấy tên cột nếu value = 1
            # Thể loại chính + số lượng --> Đếm số thể loại của mỗi phim
            self.movies['main_genre'] = self.movies[genre_cols].idxmax(axis=1)
            self.movies['genre_count'] = self.movies[genre_cols].sum(axis=1)

            if use_cache:
                ensure_dir(cache_path.parent)
                save_pickle(self.movies, cache_path)

        logger.info(f"Loaded {len(self.movies)} movies")

        return self.movies

    def load_users(self, use_cache: bool = True) -> pd.DataFrame:
        """Load thông tin user"""
        cache_path = self.data_path.parent / "processed" / "users_cache.pkl"

        if use_cache and cache_path.exists():
            logger.info("Tải user từ cache...")
            self.users = load_pickle(cache_path)
        else:
            logger.info("Tải user từ files...")

            user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
            users_path = self.data_path / "u.user"

            self.users = pd.read_csv(
                users_path, sep='|', names=user_cols, encoding='latin-1'
            )

            # Ánh xạ user IDs bằng mapping từ ratings
            if self._user_mapping is not None:
                self.users['original_id'] = self.users['user_id']
                self.users['user_id'] = self.users['user_id'].map(self._user_mapping)
                # Xóa các users không có trong mapping
                self.users = self.users.dropna(subset=['user_id'])
                self.users['user_id'] = self.users['user_id'].astype(int)

            # Xử lý age groups --> Phân loại tuổi
            self.users['age_group'] = pd.cut(
                self.users['age'],
                bins=[0, 18, 25, 35, 50, 100],
                labels=['<18', '18-25', '25-35', '35-50', '50+']
            )

            if use_cache:
                ensure_dir(cache_path.parent)
                save_pickle(self.users, cache_path)

        logger.info(f"Loaded {len(self.users)} users")

        return self.users

    def get_stats(self) -> Dict[str, Any]:
        """Lấy statistics của dataset"""
        # Load data
        if self.ratings_train is None:
            self.load_ratings()
        if self.movies is None:
            self.load_movies()
        if self.users is None:
            self.load_users()

        # Basic stats
        n_users = self.ratings_train['user_id'].nunique()
        n_movies = self.ratings_train['movie_id'].nunique()
        n_ratings_train = len(self.ratings_train)
        n_ratings_test = len(self.ratings_test)

        # Sparsity
        sparsity = 1 - (n_ratings_train / (n_users * n_movies))

        # Rating statistics
        # Tóm tắt thống kê
        rating_stats = self.ratings_train['rating'].describe()
        # Đếm tần suất + sort
        rating_dist = self.ratings_train['rating'].value_counts().sort_index()

        # User statistics
        user_rating_counts = self.ratings_train['user_id'].value_counts()
        user_stats = {
            'min_ratings': user_rating_counts.min(),
            'max_ratings': user_rating_counts.max(),
            'avg_ratings': user_rating_counts.mean(),
            'median_ratings': user_rating_counts.median()
        }

        # Movie statistics
        movie_rating_counts = self.ratings_train['movie_id'].value_counts()
        movie_stats = {
            'min_ratings': movie_rating_counts.min(),
            'max_ratings': movie_rating_counts.max(),
            'avg_ratings': movie_rating_counts.mean(),
            'median_ratings': movie_rating_counts.median()
        }

        # Genre distribution
        # Phân bố
        genre_stats = {}
        if 'genres' in self.movies.columns:
            all_genres = []
            for genres_list in self.movies['genres']:
                all_genres.extend(genres_list)
            genre_counts = pd.Series(all_genres).value_counts()
            genre_stats = genre_counts.to_dict()

        stats = {
            'dataset': {
                'n_users': n_users,
                'n_movies': n_movies,
                'n_ratings_train': n_ratings_train,
                'n_ratings_test': n_ratings_test,
                'sparsity': sparsity
            },
            'ratings': {
                'min': rating_stats['min'],
                'max': rating_stats['max'],
                'mean': rating_stats['mean'],
                'std': rating_stats['std'],
                'distribution': rating_dist.to_dict()
            },
            'users': user_stats,
            'movies': movie_stats,
            'genres': genre_stats
        }

        return stats

    def get_user_item_matrix(self, rating_type: str = 'train') -> np.ndarray:
        """Tạo ma trận User-Item cho collaborative filtering"""
        ratings = self.ratings_train if rating_type == 'train' else self.ratings_test

        if ratings is None:
            self.load_ratings()
            ratings = self.ratings_train if rating_type == 'train' else self.ratings_test

        # Lấy chiều dữ liệu
        n_users = ratings['user_id'].max() + 1
        n_movies = ratings['movie_id'].max() + 1

        # Khởi tạo matrix
        matrix = np.zeros((n_users, n_movies))
        # Fill matrix
        for _, row in ratings.iterrows():
            matrix[int(row['user_id']), int(row['movie_id'])] = row['rating']

        return matrix

    def get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID"""
        if self.movies is None:
            self.load_movies()

        movie = self.movies[self.movies['movie_id'] == movie_id]
        if len(movie) > 0:
            return movie.iloc[0]['title']
        return f"Movie {movie_id}"

    def get_movie_info(self, movie_id: int) -> Dict[str, Any]:
        """Get detailed movie information"""
        if self.movies is None:
            self.load_movies()

        movie = self.movies[self.movies['movie_id'] == movie_id]
        if len(movie) == 0:
            return {'movie_id': movie_id, 'title': f'Movie {movie_id}'}

        movie_data = movie.iloc[0]
        return {
            'movie_id': movie_id,
            'title': movie_data['title'],
            'release_date': movie_data['release_date'],
            'release_year': movie_data.get('release_year'),
            'genres': movie_data.get('genres', []),
            'main_genre': movie_data.get('main_genre'),
            'imdb_url': movie_data.get('imdb_url')
        }

    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search movies by title"""
        if self.movies is None:
            self.load_movies()

        mask = self.movies['title'].str.contains(query, case=False, na=False)
        return self.movies[mask].head(limit)

    def get_user_ratings(self, user_id: int, rating_type: str = 'train') -> pd.DataFrame:
        """Get all ratings for a specific user"""
        ratings = self.ratings_train if rating_type == 'train' else self.ratings_test

        if ratings is None:
            self.load_ratings()
            ratings = self.ratings_train if rating_type == 'train' else self.ratings_test

        # Filter user
        user_ratings = ratings[ratings['user_id'] == user_id].copy()

        # Merge with movie info
        if self.movies is not None:
            user_ratings = user_ratings.merge(
                self.movies[['movie_id', 'title', 'main_genre']],
                on='movie_id',
                how='left'
            )
        # Sort by rating
        return user_ratings.sort_values('rating', ascending=False)