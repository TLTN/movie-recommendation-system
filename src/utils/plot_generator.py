import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from .file_utils import ensure_dir

logger = logging.getLogger(__name__)


class PlotGenerator:
    """các hàm để tạo các biểu đồ riêng biệt, trực quan hóa dữ liệu và hiệu suất của hệ thống"""

    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir) # path
        ensure_dir(self.output_dir) # if null

        # style
        plt.style.use('seaborn-v0_8')
        # color
        sns.set_palette("Set2")

    def plot_rating_distribution(self, ratings_df: pd.DataFrame, # df cột rating
                                 save_name: str = None):
        """Biểu đồ phân phối rating"""
        if save_name is None:
            save_name = f"rating_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # default name

        plt.figure(figsize=(12, 8))

        # biểu đồ cột của số lượng mỗi điểm số, thêm nhãn phần trăm
        plt.subplot(2, 2, 1)
        rating_counts = ratings_df['rating'].value_counts().sort_index()
        bars = plt.bar(rating_counts.index, rating_counts.values,
                       color='skyblue', alpha=0.8, edgecolor='black')
        plt.title('Rating Distribution (Count)', fontsize=14, fontweight='bold')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)

        # Add percentage labels
        total = len(ratings_df)
        for bar, count in zip(bars, rating_counts.values):
            pct = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                     f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

        # biểu đồ tròn của phân phối điểm số với phần trăm
        plt.subplot(2, 2, 2)
        plt.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
                colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC'])
        plt.title('Rating Distribution (Percentage)', fontsize=14, fontweight='bold')

        # histogram của điểm số
        plt.subplot(2, 2, 3)
        plt.hist(ratings_df['rating'], bins=5, color='lightgreen', # 5 bin
                 alpha=0.8, edgecolor='black')
        plt.title('Rating Histogram', fontsize=14, fontweight='bold')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.3)

        # biểu đồ hộp của điểm số
        plt.subplot(2, 2, 4)
        box_plot = plt.boxplot(ratings_df['rating'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
        plt.title('Rating Box Plot', fontsize=14, fontweight='bold')
        plt.ylabel('Rating')
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout() # save bố cục

        # Save
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Rating distribution plot saved: {save_path}")
        return save_path # path

    def plot_user_activity(self, ratings_df: pd.DataFrame, save_name: str = None):
        """Biểu đồ hoạt động user"""
        if save_name is None:
            save_name = f"user_activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        plt.figure(figsize=(15, 10))

        user_activity = ratings_df['user_id'].value_counts()
        user_avg_ratings = ratings_df.groupby('user_id')['rating'].mean()

        # Phân phối số lượng đánh giá mỗi người dùng
        plt.subplot(2, 2, 1)
        plt.hist(user_activity, bins=50, color='lightblue', alpha=0.8, edgecolor='black')
        plt.title('User Activity Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Ratings per User')
        plt.ylabel('Number of Users')
        plt.axvline(user_activity.mean(), color='red', linestyle='--',
                    label=f'Mean: {user_activity.mean():.1f}')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Phân phối điểm trung bình của người dùng
        plt.subplot(2, 2, 2)
        plt.hist(user_avg_ratings, bins=30, color='lightcoral', alpha=0.8, edgecolor='black')
        plt.title('User Average Rating Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Users')
        plt.axvline(user_avg_ratings.mean(), color='red', linestyle='--',
                    label=f'Mean: {user_avg_ratings.mean():.2f}')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Biểu đồ cột của top 20 người dùng tích cực nhất
        plt.subplot(2, 2, 3)
        top_users = user_activity.head(20)
        plt.bar(range(len(top_users)), top_users.values, color='lightgreen', alpha=0.8)
        plt.title('Top 20 Most Active Users', fontsize=14, fontweight='bold')
        plt.xlabel('User Rank')
        plt.ylabel('Number of Ratings')
        plt.grid(axis='y', alpha=0.3)

        # Biểu đồ phân tán --- mối quan hệ giữa số lượng đánh giá và điểm trung bình
        plt.subplot(2, 2, 4)
        plt.scatter(user_activity, user_avg_ratings, alpha=0.6, s=30)
        plt.title('User Activity vs Average Rating', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"User activity plot saved: {save_path}")
        return save_path
    # Phân tích hành vi người dùng, đặc biệt là mức độ tích cực và xu hướng đánh giá.

    def plot_movie_popularity(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None, save_name: str = None):
        """Biểu đồ độ phổ biến phim"""
        if save_name is None:
            save_name = f"movie_popularity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        plt.figure(figsize=(15, 10))

        # số lượng đánh giá
        movie_popularity = ratings_df['movie_id'].value_counts()
        # điểm trung bình
        movie_avg_ratings = ratings_df.groupby('movie_id')['rating'].mean()

        # Phân phối số lượng đánh giá mỗi phim
        plt.subplot(2, 2, 1)
        plt.hist(movie_popularity, bins=50, color='lightsalmon', alpha=0.8, edgecolor='black')
        plt.title('Movie Popularity Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Ratings per Movie')
        plt.ylabel('Number of Movies')
        plt.axvline(movie_popularity.mean(), color='red', linestyle='--',
                    label=f'Mean: {movie_popularity.mean():.1f}')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Phân phối điểm trung bình của phim
        plt.subplot(2, 2, 2)
        plt.hist(movie_avg_ratings, bins=30, color='lightsteelblue', alpha=0.8, edgecolor='black')
        plt.title('Movie Average Rating Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Movies')
        plt.axvline(movie_avg_ratings.mean(), color='red', linestyle='--',
                    label=f'Mean: {movie_avg_ratings.mean():.2f}')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        # Biểu đồ cột của top 20 phim phổ biến nhất
        plt.subplot(2, 2, 3)
        top_movies = movie_popularity.head(20)
        bars = plt.bar(range(len(top_movies)), top_movies.values, color='lightpink', alpha=0.8)
        plt.title('Top 20 Most Popular Movies', fontsize=14, fontweight='bold')
        plt.xlabel('Movie Rank')
        plt.ylabel('Number of Ratings')
        plt.grid(axis='y', alpha=0.3)

        # Biểu đồ phân tán --- mối quan hệ giữa số lượng đánh giá và điểm trung bình
        plt.subplot(2, 2, 4)
        plt.scatter(movie_popularity, movie_avg_ratings, alpha=0.6, s=30)
        plt.title('Movie Popularity vs Average Rating', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Average Rating')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Movie popularity plot saved: {save_path}")
        return save_path
    # Phân tích xu hướng độ phổ biến của phim

    def plot_model_performance(self, results: dict, model_name: str, save_name: str = None):
        """Biểu đồ hiệu suất model"""
        if save_name is None:
            save_name = f"model_performance_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        plt.figure(figsize=(15, 10))

        rating_metrics = {}
        ranking_metrics = {}

        # phân loại chỉ số
        for key, value in results.items():
            if isinstance(value, (int, float)):
                if key in ['rmse', 'mae', 'r2_score']: # RMSE, MAE, R²
                    rating_metrics[key.upper()] = value
                elif 'at_' in key: # Precision@K, Recall@K, NDCG@K
                    ranking_metrics[key] = value

        # Biểu đồ cột của các chỉ số dự đoán điểm số
        if rating_metrics:
            plt.subplot(2, 2, 1)
            bars = plt.bar(rating_metrics.keys(), rating_metrics.values(),
                           color='skyblue', alpha=0.8, edgecolor='black')
            plt.title(f'{model_name} - Rating Prediction Metrics', fontsize=14, fontweight='bold')
            plt.ylabel('Metric Value')
            plt.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, rating_metrics.values()):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

        # Biểu đồ cột của các chỉ số xếp hạng tại K=5, 10, 20
        if ranking_metrics:
            plt.subplot(2, 2, 2)

            # Group by K values
            k_values = [5, 10, 20]
            precision_values = [ranking_metrics.get(f'precision_at_{k}', 0) for k in k_values]
            recall_values = [ranking_metrics.get(f'recall_at_{k}', 0) for k in k_values]
            ndcg_values = [ranking_metrics.get(f'ndcg_at_{k}', 0) for k in k_values]

            x = np.arange(len(k_values))
            width = 0.25

            plt.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
            plt.bar(x, recall_values, width, label='Recall', alpha=0.8)
            plt.bar(x + width, ndcg_values, width, label='NDCG', alpha=0.8)

            plt.xlabel('K Value')
            plt.ylabel('Metric Value')
            plt.title(f'{model_name} - Ranking Metrics', fontsize=14, fontweight='bold')
            plt.xticks(x, [f'K={k}' for k in k_values])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)

        # Biểu đồ radar của các chỉ số
        plt.subplot(2, 2, 3)
        all_metrics = {**rating_metrics, **{k: v for k, v in ranking_metrics.items() if 'at_10' in k}}

        if len(all_metrics) >= 3:
            metrics_names = list(all_metrics.keys())[:6]  # Max 6 metrics
            metrics_values = [all_metrics[name] for name in metrics_names]

            # Normalize values to 0-1 scale for radar chart
            max_val = max(metrics_values)
            normalized_values = [v / max_val if max_val > 0 else 0 for v in metrics_values]

            angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
            normalized_values += normalized_values[:1]  # Complete the circle
            angles += angles[:1]

            ax = plt.subplot(2, 2, 3, projection='polar')
            ax.plot(angles, normalized_values, 'o-', linewidth=2)
            ax.fill(angles, normalized_values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_names)
            ax.set_title(f'{model_name} - Overall Performance', fontsize=12, fontweight='bold')

        # Hiển thị thông tin mô hình và tham số
        plt.subplot(2, 2, 4)
        info_text = f"Model: {model_name}\n"
        if 'model_params' in results:
            params = results['model_params']
            for key, value in list(params.items())[:5]:  # Show first 5 params
                info_text += f"{key}: {value}\n"

        plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        plt.axis('off')
        plt.title('Model Information', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        logger.info(f"Model performance plot saved: {save_path}")
        return save_path