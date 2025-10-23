#             rec_items = [item_id for item_id, _ in user_recs]
#             recommendations.append(rec_items)
#
#             # Get relevant items (ratings above threshold)
#             relevant_mask = group['rating'] >= self.rating_threshold
#             user_relevant = group[relevant_mask]['item_id'].astype(int).tolist()
#             user_ratings = group[relevant_mask]['rating'].tolist()
#
#             relevant_items.append(user_relevant)
#             relevant_ratings.append(user_ratings)
#
#         results = {}
#
#         # Precision, Recall, F1 at k
#         if f'precision_at_{k}' in metrics or 'precision_at_k' in metrics:
#             results[f'precision_at_{k}'] = precision_at_k(recommendations, relevant_items, k)
#
#         if f'recall_at_{k}' in metrics or 'recall_at_k' in metrics:
#             results[f'recall_at_{k}'] = recall_at_k(recommendations, relevant_items, k)
#
#         if f'f1_at_{k}' in metrics or 'f1_at_k' in metrics:
#             results[f'f1_at_{k}'] = f1_at_k(recommendations, relevant_items, k)
#
#         # NDCG at k
#         if f'ndcg_at_{k}' in metrics or 'ndcg_at_k' in metrics:
#             results[f'ndcg_at_{k}'] = ndcg_at_k(recommendations, relevant_items, relevant_ratings, k)
#
#         # Hit rate at k
#         if f'hit_rate_at_{k}' in metrics or 'hit_rate' in metrics:
#             results[f'hit_rate_at_{k}'] = hit_rate_at_k(recommendations, relevant_items, k)
#
#         # Mean Average Precision
#         if 'map' in metrics:
#             results['map'] = mean_average_precision(recommendations, relevant_items)
#
#         logger.info(f"Ranking metrics at k={k} - Precision: {results.get(f'precision_at_{k}', 0):.4f}, "
#                     f"Recall: {results.get(f'recall_at_{k}', 0):.4f}")
#
#         return results
#
#     def _evaluate_system_metrics(self, model: BaseRecommender, test_data: np.ndarray,
#                                  train_data: Optional[np.ndarray] = None,
#                                  k: int = 10) -> Dict[str, float]:
#         """Evaluate system-level metrics"""
#         logger.info("Evaluating system-level metrics...")
#
#         results = {}
#
#         # Get all unique users from test data
#         unique_users = np.unique(test_data[:, 0].astype(int))
#
#         # Generate recommendations for all users
#         all_recommendations = []
#         user_profiles = []
#
#         for user_id in unique_users:
#             user_recs = model.get_user_recommendations(user_id, exclude_seen=True, n_recommendations=k)
#             rec_items = [item_id for item_id, _ in user_recs]
#             all_recommendations.append(rec_items)
#
#             # Get user profile from training data
#             if train_data is not None:
#                 user_profile = train_data[train_data[:, 0] == user_id, 1].astype(int).tolist()
#                 user_profiles.append(user_profile)
#             else:
#                 user_profiles.append([])
#
#         # Coverage
#         total_items = model.n_items
#         results['coverage'] = coverage(all_recommendations, total_items)
#
#         # Diversity (simple version without similarity matrix)
#         results['diversity'] = diversity(all_recommendations)
#
#         # Novelty and Serendipity require additional data processing
#         if train_data is not None:
#             # Compute item popularity
#             item_popularity = self._compute_item_popularity(train_data)
#             results['novelty'] = novelty(all_recommendations, item_popularity)
#
#             # For serendipity, we would need an item similarity matrix
#             # This is computationally expensive, so we skip it for now
#             # results['serendipity'] = serendipity(all_recommendations, user_profiles, item_similarity)
#
#         logger.info(f"System metrics - Coverage: {results['coverage']:.4f}, "
#                     f"Diversity: {results['diversity']:.4f}")
#
#         return results
#
#     def _compute_item_popularity(self, train_data: np.ndarray) -> Dict[int, float]:
#         """Compute item popularity from training data"""
#         item_counts = defaultdict(int)
#         total_interactions = len(train_data)
#
#         for _, item_id, _ in train_data:
#             item_counts[int(item_id)] += 1
#
#         # Convert to probabilities
#         item_popularity = {item_id: count / total_interactions
#                            for item_id, count in item_counts.items()}
#
#         return item_popularity
#
#     def compare_models(self, models: List[BaseRecommender], test_data: np.ndarray,
#                        train_data: Optional[np.ndarray] = None,
#                        metrics: Optional[List[str]] = None,
#                        k_values: Optional[List[int]] = None) -> pd.DataFrame:
#         """
#         Compare multiple models
#
#         Args:
#             models: List of trained models
#             test_data: Test dataset
#             train_data: Training dataset
#             metrics: List of metrics to compute
#             k_values: K values for ranking metrics
#
#         Returns:
#             DataFrame with comparison results
#         """
#         logger.info(f"Comparing {len(models)} models...")
#
#         comparison_results = []
#
#         for model in models:
#             model_results = self.evaluate_model(
#                 model, test_data, train_data, metrics, k_values
#             )
#
#             # Flatten results for comparison table
#             flat_results = {'model': model.__class__.__name__}
#
#             for key, value in model_results.items():
#                 if isinstance(value, (int, float)):
#                     flat_results[key] = value
#                 elif key == 'model_params':
#                     # Add key parameters
#                     for param_key, param_value in value.items():
#                         if isinstance(param_value, (int, float, str)):
#                             flat_results[f'param_{param_key}'] = param_value
#
#             comparison_results.append(flat_results)
#
#         comparison_df = pd.DataFrame(comparison_results)
#
#         logger.info("Model comparison completed")
#         return comparison_df
#
#     def cross_validate(self, model_class, model_params: Dict[str, Any],
#                        data: np.ndarray, n_folds: int = 5,
#                        metrics: Optional[List[str]] = None,
#                        random_state: int = 42) -> Dict[str, List[float]]:
#         """
#         Perform cross-validation on a model
#
#         Args:
#             model_class: Model class to evaluate
#             model_params: Model parameters
#             data: Full dataset
#             n_folds: Number of folds for cross-validation
#             metrics: List of metrics to compute
#             random_state: Random state for reproducibility
#
#         Returns:
#             Dictionary with lists of metric values across folds
#         """
#         logger.info(f"Performing {n_folds}-fold cross-validation...")
#
#         np.random.seed(random_state)
#
#         # Shuffle data
#         shuffled_indices = np.random.permutation(len(data))
#         shuffled_data = data[shuffled_indices]
#
#         # Create folds
#         fold_size = len(data) // n_folds
#         folds = []
#
#         for i in range(n_folds):
#             start_idx = i * fold_size
#             end_idx = start_idx + fold_size if i < n_folds - 1 else len(data)
#             folds.append(shuffled_data[start_idx:end_idx])
#
#         cv_results = defaultdict(list)
#
#         for fold_idx in range(n_folds):
#             logger.info(f"Evaluating fold {fold_idx + 1}/{n_folds}")
#
#             # Create train/test split
#             test_fold = folds[fold_idx]
#             train_folds = [folds[i] for i in range(n_folds) if i != fold_idx]
#             train_data = np.vstack(train_folds)
#
#             # Train model
#             model = model_class(**model_params)
#             model.fit(train_data)
#
#             # Evaluate model
#             fold_results = self.evaluate_model(model, test_fold, train_data, metrics)
#
#             # Store results
#             for metric, value in fold_results.items():
#                 if isinstance(value, (int, float)):
#                     cv_results[metric].append(value)
#
#         logger.info("Cross-validation completed")
#         return dict(cv_results)
#
#     def statistical_significance_test(self, results1: List[float], results2: List[float],
#                                       test_type: str = 'paired_t') -> Dict[str, float]:
#         """
#         Test statistical significance between two sets of results
#
#         Args:
#             results1: Results from model 1
#             results2: Results from model 2
#             test_type: Type of statistical test ('paired_t', 'wilcoxon')
#
#         Returns:
#             Dictionary with test statistics and p-value
#         """
#         try:
#             from scipy import stats
#         except ImportError:
#             logger.warning("SciPy not available for statistical tests")
#             return {'p_value': None, 'statistic': None}
#
#         if len(results1) != len(results2):
#             raise ValueError("Results lists must have the same length")
#
#         if test_type == 'paired_t':
#             statistic, p_value = stats.ttest_rel(results1, results2)
#         elif test_type == 'wilcoxon':
#             statistic, p_value = stats.wilcoxon(results1, results2)
#         else:
#             raise ValueError(f"Unknown test type: {test_type}")
#
#         return {
#             'test_type': test_type,
#             'statistic': float(statistic),
#             'p_value': float(p_value),
#             'significant': p_value < 0.05
#         }
#
#     def learning_curve(self, model_class, model_params: Dict[str, Any],
#                        train_data: np.ndarray, test_data: np.ndarray,
#                        train_sizes: Optional[List[float]] = None) -> Dict[str, List[float]]:
#         """
#         Generate learning curve for a model
#
#         Args:
#             model_class: Model class to evaluate
#             model_params: Model parameters
#             train_data: Training dataset
#             test_data: Test dataset
#             train_sizes: Fractions of training data to use
#
#         Returns:
#             Dictionary with training sizes and corresponding performance
#         """
#         logger.info("Generating learning curve...")
#
#         train_sizes = train_sizes or [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
#
#         results = {
#             'train_sizes': [],
#             'train_rmse': [],
#             'test_rmse': [],
#             'train_time': []
#         }
#
#         for size in train_sizes:
#             logger.info(f"Training with {size * 100:.1f}% of data")
#
#             # Sample training data
#             n_samples = int(len(train_data) * size)
#             sampled_indices = np.random.choice(len(train_data), n_samples, replace=False)
#             sampled_train_data = train_data[sampled_indices]
#
#             # Train model
#             start_time = time.time()
#             model = model_class(**model_params)
#             model.fit(sampled_train_data)
#             train_time = time.time() - start_time
#
#             # Evaluate on train and test
#             train_results = self._evaluate_rating_prediction(model, sampled_train_data)
#             test_results = self._evaluate_rating_prediction(model, test_data)
#
#             # Store results
#             results['train_sizes'].append(n_samples)
#             results['train_rmse'].append(train_results['rmse'])
#             results['test_rmse'].append(test_results['rmse'])
#             results['train_time'].append(train_time)
#
#         logger.info("Learning curve generation completed")
#         return results
#
#     def parameter_sensitivity(self, model_class, base_params: Dict[str, Any],
#                               param_ranges: Dict[str, List[Any]],
#                               train_data: np.ndarray, test_data: np.ndarray,
#                               metric: str = 'rmse') -> Dict[str, Dict[str, List[float]]]:
#         """
#         Analyze parameter sensitivity
#
#         Args:
#             model_class: Model class to evaluate
#             base_params: Base model parameters
#             param_ranges: Dictionary of parameter names and their ranges to test
#             train_data: Training dataset
#             test_data: Test dataset
#             metric: Metric to optimize
#
#         Returns:
#             Dictionary with parameter sensitivity results
#         """
#         logger.info("Analyzing parameter sensitivity...")
#
#         sensitivity_results = {}
#
#         for param_name, param_values in param_ranges.items():
#             logger.info(f"Testing parameter: {param_name}")
#
#             param_results = {
#                 'values': [],
#                 'scores': []
#             }
#
#             for param_value in param_values:
#                 # Create model with modified parameter
#                 model_params = base_params.copy()
#                 model_params[param_name] = param_value
#
#                 try:
#                     # Train and evaluate model
#                     model = model_class(**model_params)
#                     model.fit(train_data)
#
#                     results = self.evaluate_model(model, test_data, metrics=[metric])
#                     score = results.get(metric, float('inf'))
#
#                     param_results['values'].append(param_value)
#                     param_results['scores'].append(score)
#
#                     logger.info(f"  {param_name}={param_value}: {metric}={score:.4f}")
#
#                 except Exception as e:
#                     logger.warning(f"  {param_name}={param_value}: Error - {e}")
#                     param_results['values'].append(param_value)
#                     param_results['scores'].append(float('inf'))
#
#             sensitivity_results[param_name] = param_results
#
#         logger.info("Parameter sensitivity analysis completed")
#         return sensitivity_results
#
#     def get_summary_report(self, model_name: str = None) -> str:
#         """
#         Generate a summary report of evaluation results
#
#         Args:
#             model_name: Specific model name to report on (if None, reports all)
#
#         Returns:
#             Formatted summary report string
#         """
#         if not self.results:
#             return "No evaluation results available."
#
#         report_lines = ["=" * 50]
#         report_lines.append("RECOMMENDATION SYSTEM EVALUATION REPORT")
#         report_lines.append("=" * 50)
#
#         models_to_report = [model_name] if model_name and model_name in self.results else self.results.keys()
#
#         for model in models_to_report:
#             if model not in self.results:
#                 continue
#
#             results = self.results[model]
#             report_lines.append(f"\nModel: {model}")
#             report_lines.append("-" * 30)
#
#             # Rating prediction metrics
#             rating_metrics = ['rmse', 'mae', 'r2_score']
#             if any(metric in results for metric in rating_metrics):
#                 report_lines.append("Rating Prediction:")
#                 for metric in rating_metrics:
#                     if metric in results:
#                         report_lines.append(f"  {metric.upper()}: {results[metric]:.4f}")
#
#             # Ranking metrics
#             ranking_metrics = [k for k in results.keys() if 'at_' in k or k in ['map']]
#             if ranking_metrics:
#                 report_lines.append("\nRanking Performance:")
#                 for metric in sorted(ranking_metrics):
#                     report_lines.append(f"  {metric}: {results[metric]:.4f}")
#
#             # System metrics
#             system_metrics = ['coverage', 'diversity', 'novelty']
#             if any(metric in results for metric in system_metrics):
#                 report_lines.append("\nSystem Metrics:")
#                 for metric in system_metrics:
#                     if metric in results:
#                         report_lines.append(f"  {metric.capitalize()}: {results[metric]:.4f}")
#
#         return "\n".join(report_lines)
#
#     def export_results(self, filepath: str, format: str = 'json') -> None:
#         """
#         Export evaluation results to file
#
#         Args:
#             filepath: Output file path
#             format: Export format ('json', 'csv')
#         """
#         import json
#         from ..utils.file_utils import ensure_dir
#
#         ensure_dir(Path(filepath).parent)
#
#         if format == 'json':
#             with open(filepath, 'w') as f:
#                 json.dump(self.results, f, indent=2, default=str)
#         elif format == 'csv':
#             # Flatten results for CSV export
#             flat_results = []
#             for model_name, results in self.results.items():
#                 flat_result = {'model': model_name}
#                 for key, value in results.items():
#                     if isinstance(value, (int, float)):
#                         flat_result[key] = value
#                 flat_results.append(flat_result)
#
#             df = pd.DataFrame(flat_results)
#             df.to_csv(filepath, index=False)
#         else:
#             raise ValueError(f"Unsupported format: {format}")
#
#         logger.info(f"Results exported to {filepath}")


import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from .metrics import rmse, mae, precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k
from ..models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {} # từ điển cấu hình
        self.results = {} # init
        self.rating_threshold = self.config.get('rating_threshold', 3.5) # set threshold
    # Chuẩn bị đối tượng để đánh giá mô hình, với ngưỡng điểm số để xác định mục liên quan

    def evaluate_model(self,
                       model: BaseRecommender, # model
                       test_data: np.ndarray, # test
                       train_data: Optional[np.ndarray] = None, # train
                       metrics: Optional[List[str]] = None, # RMSE, MAE, Precision@K, Recall@K, NDCG@K, Hit Rate
                       k_values: Optional[List[int]] = None) -> Dict[str, Any]: # 5, 10, 20
        logger.info("Starting model evaluation...")

        if not model.is_fitted: # đã train?
            raise ValueError("Model must be fitted before evaluation")

        # set metrics, k_values
        metrics = metrics or ['rmse', 'mae', 'precision_at_k', 'recall_at_k', 'ndcg_at_k', 'hit_rate']
        k_values = k_values or [5, 10, 20]

        # tạo dictionary = model + para
        results = {
            'model_name': model.__class__.__name__,
            'model_params': model.get_params()
        }

        # nếu có rmse, mae trong metrics
        if any(metric in ['rmse', 'mae'] for metric in metrics):
            rating_results = self._evaluate_rating_prediction(model, test_data) # call hàm
            results.update(rating_results)


        ranking_metrics = [m for m in metrics if 'at_k' in m or m in ['hit_rate']]
        # nếu có ranking
        if ranking_metrics:
            for k in k_values:
                ranking_results = self._evaluate_ranking(model, test_data, k, ranking_metrics)
                results.update(ranking_results)

        self.results[model.__class__.__name__] = results # save
        logger.info("Model evaluation completed")
        return results
    # Đánh giá tổng thể hiệu suất mô hình, gồm dự đoán điểm số và xếp hạng gợi ý

    def _evaluate_rating_prediction(self, model: BaseRecommender, test_data: np.ndarray) -> Dict[str, float]:
        logger.info("Evaluating rating prediction...")

        y_true = [] # điểm thực tế
        y_pred = [] # điểm dự đoán

        for user_id, item_id, rating in test_data:
            user_id, item_id = int(user_id), int(item_id)
            predicted_rating = model.predict(user_id, item_id) # predict điểm số

            y_true.append(rating)
            y_pred.append(predicted_rating)

        # chuyển mảng
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Tính rmse, mae
        results = {
            'rmse': rmse(y_true, y_pred),
            'mae': mae(y_true, y_pred)
        }

        logger.info(f"RMSE: {results['rmse']:.4f}, MAE: {results['mae']:.4f}")
        return results
    # Đánh giá độ chính xác dự đoán điểm số

    def _evaluate_ranking(self, model: BaseRecommender, test_data: np.ndarray,
                          k: int, metrics: List[str]) -> Dict[str, float]:
        """Đánh giá chất lượng xếp hạng gợi ý của mô hình tại giá trị K cụ thể"""
        logger.info(f"Evaluating ranking metrics at k={k}...")

        # chuyển test_data --> DataFrame
        test_df = pd.DataFrame(test_data, columns=['user_id', 'item_id', 'rating'])
        # nhóm theo user_id
        user_groups = test_df.groupby('user_id')

        recommendations = []
        relevant_items = []

        for user_id, group in user_groups:
            user_id = int(user_id)

            # model.get_user_recommendations + exclude_seen=True --> lấy top-K gợi ý
            user_recs = model.get_user_recommendations(user_id, exclude_seen=True, n_recommendations=k)
            # trích xuất item_id
            rec_items = [item_id for item_id, _ in user_recs]
            recommendations.append(rec_items)

            # lọc các mục liên quan
            relevant_mask = group['rating'] >= self.rating_threshold # rating >= rating_threshold
            user_relevant = group[relevant_mask]['item_id'].astype(int).tolist()
            relevant_items.append(user_relevant)

        results = {}

        # Metrics
        if f'precision_at_{k}' in metrics or 'precision_at_k' in metrics:
            results[f'precision_at_{k}'] = precision_at_k(recommendations, relevant_items, k)

        if f'recall_at_{k}' in metrics or 'recall_at_k' in metrics:
            results[f'recall_at_{k}'] = recall_at_k(recommendations, relevant_items, k)

        if f'ndcg_at_{k}' in metrics or 'ndcg_at_k' in metrics:
            results[f'ndcg_at_{k}'] = ndcg_at_k(recommendations, relevant_items, k)

        if f'hit_rate_at_{k}' in metrics or 'hit_rate' in metrics:
            results[f'hit_rate_at_{k}'] = hit_rate_at_k(recommendations, relevant_items, k)

        logger.info(f"Ranking metrics at k={k} completed")
        return results
    # Đánh giá chất lượng danh sách gợi ý (KNN, NCF)