import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Plotly not available. Interactive plots will not work.")

from .file_utils import ensure_dir

logger = logging.getLogger(__name__)


class MovieRecommenderVisualizer:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # kiểu biểu đồ
        plt.style.use(self.config.get('style', 'seaborn-v0_8'))
        # bảng màu
        sns.set_palette(self.config.get('color_palette', 'Set2'))

        # size, dpi, format
        self.figsize = tuple(self.config.get('figure_size', [12, 8]))
        self.dpi = self.config.get('dpi', 300)
        self.save_format = self.config.get('save_format', 'png')

        # out dir
        self.output_dir = Path(self.config.get('output_dir', 'results/plots'))
        ensure_dir(self.output_dir)

    def plot_data_distribution(self, data_stats: Dict[str, Any],
                               save_path: Optional[str] = None) -> None:
        """Vẽ biểu đồ phân bố dữ liệu"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi) # lưới 2x2 subplot

        # phân bố điểm
        if 'ratings' in data_stats and 'distribution' in data_stats['ratings']:
            rating_dist = data_stats['ratings']['distribution']
            ratings = list(rating_dist.keys())
            counts = list(rating_dist.values())

            axes[0, 0].bar(ratings, counts, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Rating Distribution')
            axes[0, 0].set_xlabel('Rating')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True, alpha=0.3)

        # Hoạt động user
        axes[0, 1].hist(np.random.exponential(10, 1000), bins=50, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('User Activity Distribution')
        axes[0, 1].set_xlabel('Number of Ratings per User')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].grid(True, alpha=0.3)

        # độ phổ biến phim
        axes[1, 0].hist(np.random.exponential(5, 1000), bins=50, color='salmon', alpha=0.7)
        axes[1, 0].set_title('Item Popularity Distribution')
        axes[1, 0].set_xlabel('Number of Ratings per Item')
        axes[1, 0].set_ylabel('Number of Items')
        axes[1, 0].grid(True, alpha=0.3)

        # dataset overview
        if 'dataset' in data_stats:
            dataset_info = data_stats['dataset']
            labels = ['Users', 'Items', 'Ratings (Train)', 'Ratings (Test)']
            values = [
                dataset_info.get('n_users', 0),
                dataset_info.get('n_movies', 0),
                dataset_info.get('n_ratings_train', 0),
                dataset_info.get('n_ratings_test', 0)
            ]

            axes[1, 1].bar(range(len(labels)), values, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
            axes[1, 1].set_title('Dataset Overview')
            axes[1, 1].set_xticks(range(len(labels)))
            axes[1, 1].set_xticklabels(labels, rotation=45)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Data distribution plot saved to {save_path}")

        plt.show()

    def plot_model_comparison(self,
                              comparison_df: pd.DataFrame, # df chứa data
                              metrics: List[str] = None,
                              save_path: Optional[str] = None) -> None:
        """So sánh hiệu suất các mô hình"""
        if metrics is None:
            # lấy các cột số từ comparison_df, bỏ qua evaluation_time
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
            metrics = [col for col in numeric_cols if col not in ['evaluation_time']]

        # n_rows, n_cols dựa trên số lượng chỉ số, tối đa 3 cột
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=self.dpi) # lưới subplot với kích thước động
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            if metric in comparison_df.columns:
                values = comparison_df[metric].values
                models = comparison_df['model'].values

                # biểu đồ cột với giá trị, mô hình
                bars = ax.bar(models, values, alpha=0.7)
                ax.set_title(f'{metric.upper()}')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)

                # Thêm nhãn
                for bar, value in zip(bars, values):
                    if not np.isnan(value):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')

        # Ẩn subplot trống
        for i in range(len(metrics), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()
    # So sánh hiệu suất các mô hình dựa trên các chỉ số từ RecommenderEvaluator

    def plot_learning_curve(self, learning_curve_data: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """Vẽ đường cong học tập"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        train_sizes = learning_curve_data['train_sizes']
        train_rmse = learning_curve_data['train_rmse']
        test_rmse = learning_curve_data['test_rmse']
        train_time = learning_curve_data['train_time']

        # 2 subplot
        # đường cong RMSE train/test
        ax1.plot(train_sizes, train_rmse, 'o-', label='Train RMSE', color='blue', alpha=0.7)
        ax1.plot(train_sizes, test_rmse, 'o-', label='Test RMSE', color='red', alpha=0.7)
        ax1.set_xlabel('Training Set Size')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Learning Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # thời gian huấn luyện
        ax2.plot(train_sizes, train_time, 'o-', color='green', alpha=0.7)
        ax2.set_xlabel('Training Set Size')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time vs Dataset Size')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Learning curve plot saved to {save_path}")

        plt.show()
    # Phân tích hiệu suất mô hình khi tăng kích thước dữ liệu huấn luyện

    def plot_parameter_sensitivity(self, sensitivity_data: Dict[str, Dict[str, List[float]]],
                                   save_path: Optional[str] = None) -> None:
        """Phân tích độ nhạy của tham số mô hình"""
        n_params = len(sensitivity_data)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), dpi=self.dpi) # lưới subplot động
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, (param_name, param_data) in enumerate(sensitivity_data.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            values = param_data['values']
            scores = param_data['scores']

            # biểu đồ đường của giá trị tham số so với điểm số (RMSE).
            finite_mask = np.isfinite(scores)
            if np.any(finite_mask):
                filtered_values = np.array(values)[finite_mask]
                filtered_scores = np.array(scores)[finite_mask]

                ax.plot(filtered_values, filtered_scores, 'o-', alpha=0.7)
                ax.set_xlabel(param_name)
                ax.set_ylabel('RMSE')
                ax.set_title(f'Sensitivity: {param_name}')
                ax.grid(True, alpha=0.3)

                # giá trị tham số tốt nhất
                best_idx = np.argmin(filtered_scores)
                ax.axvline(filtered_values[best_idx], color='red', linestyle='--', alpha=0.7,
                           label=f'Best: {filtered_values[best_idx]}')
                ax.legend()

        # Ẩn subplot trống
        for i in range(len(sensitivity_data), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Parameter sensitivity plot saved to {save_path}")

        plt.show()

    def plot_training_history(self, training_history: Dict[str, List[float]],
                              save_path: Optional[str] = None) -> None:
        """Vẽ đường cong loss trong quá trình huấn luyện"""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        epochs = range(1, len(training_history['train_losses']) + 1)

        ax.plot(epochs, training_history['train_losses'], 'b-', label='Training Loss', alpha=0.7)

        if 'val_losses' in training_history and training_history['val_losses']:
            ax.plot(epochs, training_history['val_losses'], 'r-', label='Validation Loss', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()

    def plot_recommendation_analysis(self,
                                     user_id: int,
                                     recommendations: List[Tuple[int, float]], # Danh sách tuple -- model.get_user_recommendations
                                     user_history: List[Tuple[int, float, str]] = None, # Danh sách tuple -- tùy chọn
                                     movie_titles: Dict[int, str] = None, # Từ điển ánh xạ movie_id sang tiêu đề phim
                                     save_path: Optional[str] = None) -> None:
        """Vẽ biểu đồ phân tích gợi ý cho một người dùng, bao gồm top gợi ý và lịch sử đánh giá"""
        # Tạo hai subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # Recommendation scores
        # Vẽ biểu đồ cột ngang cho top 10
        if recommendations:
            movie_ids, scores = zip(*recommendations[:10])  # Top 10

            if movie_titles:
                labels = [movie_titles.get(mid, f'Movie {mid}')[:30] for mid in movie_ids]
            else:
                labels = [f'Movie {mid}' for mid in movie_ids]

            y_pos = np.arange(len(labels))

            bars = ax1.barh(y_pos, scores, alpha=0.7, color='skyblue')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels)
            ax1.set_xlabel('Predicted Rating')
            ax1.set_title(f'Top Recommendations for User {user_id}')
            ax1.grid(True, alpha=0.3)

            # Add score labels
            for bar, score in zip(bars, scores):
                ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{score:.2f}', va='center')

        # User history
        if user_history:
            movie_ids, ratings, genres = zip(*user_history[:10])  # Recent 10

            if movie_titles:
                labels = [movie_titles.get(mid, f'Movie {mid}')[:30] for mid in movie_ids]
            else:
                labels = [f'Movie {mid}' for mid in movie_ids]

            y_pos = np.arange(len(labels))

            # Mã hóa màu cho lịch sử đánh giá
            colors = ['red' if r < 3 else 'yellow' if r < 4 else 'green' for r in ratings]
            bars = ax2.barh(y_pos, ratings, alpha=0.7, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels)
            ax2.set_xlabel('User Rating')
            ax2.set_title(f'Rating History for User {user_id}')
            ax2.set_xlim(0, 5)
            ax2.grid(True, alpha=0.3)

            # Add rating labels
            for bar, rating in zip(bars, ratings):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                         f'{rating:.1f}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Recommendation analysis plot saved to {save_path}")

        plt.show()
    # Phân tích chất lượng gợi ý của mô hình cho một người dùng cụ thể

    def create_interactive_dashboard(self, data_stats: Dict[str, Any],
                                     model_results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> None:
        """Tạo dashboard tương tác bằng Plotly để trực quan hóa dữ liệu và kết quả mô hình"""
        if not PLOTLY_AVAILABLE: # check
            logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return

        # lưới 2x2 subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rating Distribution', 'Model Comparison',
                            'Dataset Overview', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )

        # Thêm các biểu đồ: phân bố điểm số, so sánh mô hình, tổng quan tập dữ liệu, và scatter Precision vs Recall
        # Rating distribution
        if 'ratings' in data_stats and 'distribution' in data_stats['ratings']:
            rating_dist = data_stats['ratings']['distribution']
            fig.add_trace(
                go.Bar(x=list(rating_dist.keys()), y=list(rating_dist.values()),
                       name="Ratings", marker_color='skyblue'),
                row=1, col=1
            )

        # Model comparison (placeholder)
        models = ['KNN', 'Matrix Factorization', 'Neural CF']
        rmse_values = [0.95, 0.89, 0.87]  # Example values
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name="RMSE", marker_color='lightgreen'),
            row=1, col=2
        )

        # Dataset overview
        if 'dataset' in data_stats:
            dataset_info = data_stats['dataset']
            labels = ['Users', 'Movies', 'Train Ratings', 'Test Ratings']
            values = [
                dataset_info.get('n_users', 0),
                dataset_info.get('n_movies', 0),
                dataset_info.get('n_ratings_train', 0),
                dataset_info.get('n_ratings_test', 0)
            ]
            fig.add_trace(
                go.Bar(x=labels, y=values, name="Counts",
                       marker_color=['blue', 'green', 'red', 'orange']),
                row=2, col=1
            )

        # Performance metrics scatter
        precision_values = [0.15, 0.18, 0.22]  # Example values
        recall_values = [0.12, 0.16, 0.19]
        fig.add_trace(
            go.Scatter(x=precision_values, y=recall_values,
                       mode='markers+text', text=models,
                       textposition="top center", name="Precision vs Recall",
                       marker=dict(size=10, color=['red', 'green', 'blue'])),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            height=800,
            title_text="Movie Recommendation System Dashboard",
            showlegend=False
        )

        # Update axes labels
        fig.update_xaxes(title_text="Rating", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        fig.update_xaxes(title_text="Category", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Precision", row=2, col=2)
        fig.update_yaxes(title_text="Recall", row=2, col=2)

        if save_path:
            fig.write_html(save_path) # save - HTML
            logger.info(f"Interactive dashboard saved to {save_path}")
        else:
            fig.show()

    def plot_genre_analysis(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
        """Phân tích thể loại phim"""
        # Tạo hai subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # số lượng phim theo thể loại
        if 'main_genre' in movies_df.columns:
            genre_counts = movies_df['main_genre'].value_counts().head(10) # Vẽ biểu đồ cột ngang cho top 10
            ax1.barh(range(len(genre_counts)), genre_counts.values, alpha=0.7)
            ax1.set_yticks(range(len(genre_counts)))
            ax1.set_yticklabels(genre_counts.index)
            ax1.set_xlabel('Number of Movies')
            ax1.set_title('Movies by Genre')
            ax1.grid(True, alpha=0.3)

        # điểm trung bình theo thể loại
        if 'main_genre' in movies_df.columns and not ratings_df.empty:
            merged_df = ratings_df.merge(movies_df[['movie_id', 'main_genre']], on='movie_id', how='left')
            genre_ratings = merged_df.groupby('main_genre')['rating'].mean().sort_values(ascending=False).head(10)
            # Vẽ biểu đồ cột ngang cho top 10
            ax2.bar(range(len(genre_ratings)), genre_ratings.values, alpha=0.7, color='lightcoral')
            ax2.set_xticks(range(len(genre_ratings)))
            ax2.set_xticklabels(genre_ratings.index, rotation=45)
            ax2.set_ylabel('Average Rating')
            ax2.set_title('Average Rating by Genre')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Genre analysis plot saved to {save_path}")

        plt.show()
    # Phân tích xu hướng thể loại trong dữ liệu phim

    def plot_user_item_heatmap(self, user_item_matrix: np.ndarray,
                               max_users: int = 50, max_items: int = 50,
                               save_path: Optional[str] = None) -> None:
        """Plot user-item interaction heatmap"""
        # Lấy mẫu ma trận với max_users, max_items
        sampled_matrix = user_item_matrix[:max_users, :max_items]

        plt.figure(figsize=self.figsize, dpi=self.dpi)

        # Create heatmap
        sns.heatmap(sampled_matrix, cmap='YlOrRd', cbar_kws={'label': 'Rating'},
                    xticklabels=False, yticklabels=False)

        plt.title(f'User-Item Rating Matrix ({max_users} users × {max_items} items)')
        plt.xlabel('Movies')
        plt.ylabel('Users')

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"User-item heatmap saved to {save_path}")

        plt.show()

    def plot_sparsity_analysis(self, user_item_matrix: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """Plot sparsity analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # User activity distribution
        user_activity = np.sum(user_item_matrix > 0, axis=1)
        ax1.hist(user_activity, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Rated Items per User')
        ax1.set_ylabel('Number of Users')
        ax1.set_title('User Activity Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(user_activity), color='red', linestyle='--',
                    label=f'Mean: {np.mean(user_activity):.1f}') # giá trị trung bình
        ax1.legend()

        # Item popularity distribution
        item_popularity = np.sum(user_item_matrix > 0, axis=0)
        ax2.hist(item_popularity, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Number of Ratings per Item')
        ax2.set_ylabel('Number of Items')
        ax2.set_title('Item Popularity Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(item_popularity), color='red', linestyle='--',
                    label=f'Mean: {np.mean(item_popularity):.1f}') # giá trị trung bình
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Sparsity analysis plot saved to {save_path}")

        plt.show()
    # Đánh giá độ thưa thớt của dữ liệu

    def plot_rating_matrix_svd(self,
                               U: np.ndarray, # ma trận người dùng
                               s: np.ndarray, # giá trị suy biến
                               Vt: np.ndarray, # ma trận phim từ SVD
                               n_components: int = 20, save_path: Optional[str] = None) -> None:
        """Plot SVD analysis of rating matrix"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=self.dpi)

        # giá trị suy biến
        ax1.plot(s[:n_components], 'o-', alpha=0.7)
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Values')
        ax1.grid(True, alpha=0.3)

        # phương sai giải thích tích lũy
        explained_variance_ratio = (s ** 2) / np.sum(s ** 2)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        ax2.plot(cumulative_variance[:n_components], 'o-', alpha=0.7, color='green')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Explained Variance')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0.9, color='red', linestyle='--', label='90%')
        ax2.legend()

        # heatmap của các nhân tố tiềm ẩn người dùng
        im = ax3.imshow(U[:50, :10], cmap='RdBu', aspect='auto')
        ax3.set_xlabel('Latent Factor')
        ax3.set_ylabel('User')
        ax3.set_title('User Latent Factors')
        plt.colorbar(im, ax=ax3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"SVD analysis plot saved to {save_path}")

        plt.show()
    # Hiểu cấu trúc dữ liệu của ma trận điểm số

    def save_all_plots(self, data_stats: Dict[str, Any],
                       evaluation_results: Dict[str, Any],
                       movies_df: pd.DataFrame = None,
                       ratings_df: pd.DataFrame = None) -> None:
        """Save all visualization plots to files"""
        logger.info("Generating and saving all visualization plots...")

        # Data distribution
        self.plot_data_distribution(
            data_stats,
            save_path=self.output_dir / f'data_distribution.{self.save_format}'
        )

        # Model comparison (if multiple models)
        if len(evaluation_results) > 1:
            comparison_data = []
            for model_name, results in evaluation_results.items():
                flat_result = {'model': model_name}
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        flat_result[key] = value
                comparison_data.append(flat_result)

            comparison_df = pd.DataFrame(comparison_data)
            self.plot_model_comparison(
                comparison_df,
                save_path=self.output_dir / f'model_comparison.{self.save_format}'
            )

        # Genre analysis (if data available)
        if movies_df is not None and ratings_df is not None:
            self.plot_genre_analysis(
                movies_df, ratings_df,
                save_path=self.output_dir / f'genre_analysis.{self.save_format}'
            )

        # Interactive dashboard
        if PLOTLY_AVAILABLE:
            self.create_interactive_dashboard(
                data_stats, evaluation_results,
                save_path=self.output_dir / 'interactive_dashboard.html'
            )

        logger.info(f"All plots saved to {self.output_dir}")

    def create_evaluation_report_plots(self,
                                       evaluator_results: Dict[str, Any], # từ điển kết quả từ RecommenderEvaluator
                                       save_dir: Optional[str] = None) -> None:
        """Create comprehensive evaluation report with plots"""
        if save_dir:
            save_dir = Path(save_dir)
            ensure_dir(save_dir)
        else:
            save_dir = self.output_dir / 'evaluation_report'
            ensure_dir(save_dir)

        logger.info(f"Creating evaluation report plots in {save_dir}")

        models = list(evaluator_results.keys())
        # Tạo từ điển metrics_data để nhóm các chỉ số theo mô hình
        metrics_data = {}

        for model_name, results in evaluator_results.items():
            for metric, value in results.items():
                if isinstance(value, (int, float)) and metric not in ['evaluation_time']:
                    if metric not in metrics_data:
                        metrics_data[metric] = {}
                    metrics_data[metric][model_name] = value

        # Vẽ biểu đồ cột cho mỗi chỉ số
        for metric, model_values in metrics_data.items():
            if len(model_values) > 1:
                plt.figure(figsize=(10, 6), dpi=self.dpi)

                models = list(model_values.keys())
                values = list(model_values.values())

                bars = plt.bar(models, values, alpha=0.7)
                plt.title(f'Model Comparison: {metric.upper()}')
                plt.ylabel(metric.upper())
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)

                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                             f'{value:.4f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(save_dir / f'{metric}_comparison.{self.save_format}',
                            dpi=self.dpi, bbox_inches='tight')
                plt.close()

        logger.info(f"Evaluation report plots saved to {save_dir}")
    # Tạo báo cáo chi tiết để so sánh mô hình