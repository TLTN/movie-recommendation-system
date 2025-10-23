import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from ..utils.file_utils import ensure_dir

logger = logging.getLogger(__name__)


class ResultsSaver:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        # crete dir
        ensure_dir(self.results_dir / "metrics")
        ensure_dir(self.results_dir / "plots")
        ensure_dir(self.results_dir / "reports")

    def save_evaluation_results(self, results: dict, model_name: str):
        # result: từ điển chứa kết quả đánh giá

        # Tạo timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Lưu JSON chi tiết
        json_file = self.results_dir / "metrics" / f"{model_name}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str) # xử lý các kiểu dữ liệu không phải JSON

        # 2. Lưu CSV summary
        csv_file = self.results_dir / "metrics" / f"{model_name}_{timestamp}_summary.csv"
        summary_data = []

        for key, value in results.items():
            if isinstance(value, (int, float)): # lấy key-value int, float
                summary_data.append({
                    'metric': key, # tên chỉ số
                    'value': value, # giá trị chỉ số
                    'model': model_name, # model
                    'timestamp': timestamp # time
                })

        df = pd.DataFrame(summary_data) # Chuyển summary_data thành DataFrame
        df.to_csv(csv_file, index=False, encoding='utf-8') # lưu CSV

        # 3. Tạo và lưu báo cáo text
        txt_file = self.results_dir / "reports" / f"{model_name}_{timestamp}_report.txt"
        self._save_text_report(results, model_name, txt_file)

        logger.info(f"Results saved:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  CSV: {csv_file}")
        logger.info(f"  Report: {txt_file}")

        return json_file, csv_file, txt_file # 3 dir

    def _save_text_report(self, results: dict, model_name: str, file_path: Path):
        """Tạo báo cáo text đẹp"""

        with open(file_path, 'w', encoding='utf-8') as f: # Mở tệp file_path ghi mã hóa UTF-8
            f.write("=" * 80 + "\n")
            f.write(f"MOVIE RECOMMENDATION SYSTEM - EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Parameters
            if 'model_params' in results:
                f.write("MODEL PARAMETERS:\n")
                f.write("-" * 40 + "\n")
                for param, value in results['model_params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")

            # chỉ số dự đoán điểm số
            rating_metrics = ['rmse', 'mae', 'r2_score']
            f.write("RATING PREDICTION METRICS:\n")
            f.write("-" * 40 + "\n")
            for metric in rating_metrics:
                if metric in results: # check
                    f.write(f"  {metric.upper()}: {results[metric]:.4f}\n")
            f.write("\n")

            # chỉ số xếp hạng
            f.write("RANKING METRICS:\n")
            f.write("-" * 40 + "\n")
            for k in [5, 10, 20]:
                precision_key = f'precision_at_{k}'
                recall_key = f'recall_at_{k}'
                ndcg_key = f'ndcg_at_{k}'
                hit_rate_key = f'hit_rate_at_{k}'

                # Kiểm tra và ghi các chỉ số precision_at_{k}, recall_at_{k}, ndcg_at_{k}, hit_rate_at_{k} nếu có trong results
                if precision_key in results:
                    f.write(f"  Precision@{k}: {results[precision_key]:.4f}\n")
                if recall_key in results:
                    f.write(f"  Recall@{k}: {results[recall_key]:.4f}\n")
                if ndcg_key in results:
                    f.write(f"  NDCG@{k}: {results[ndcg_key]:.4f}\n")
                if hit_rate_key in results:
                    f.write(f"  Hit Rate@{k}: {results[hit_rate_key]:.4f}\n")
                f.write("\n")

            # chỉ số hệ thống
            system_metrics = ['coverage', 'diversity', 'novelty'] # check coverage, diversity, novelty
            if any(metric in results for metric in system_metrics):
                f.write("SYSTEM METRICS:\n")
                f.write("-" * 40 + "\n")
                for metric in system_metrics:
                    if metric in results:
                        f.write(f"  {metric.capitalize()}: {results[metric]:.4f}\n")

    def load_all_results(self) -> pd.DataFrame:
        """Tải tất cả kết quả đánh giá từ các tệp CSV trong metrics, hợp nhất DataFrame"""
        # Tìm các tệp CSV đuôi _summary.csv trong results_dir/metrics bằng glob
        csv_files = list((self.results_dir / "metrics").glob("*_summary.csv"))

        if not csv_files:
            return pd.DataFrame() # frame rỗng

        all_results = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file) # Đọc tệp vào DataFrame
            all_results.append(df) # Thêm DataFrame vào danh sách

        return pd.concat(all_results, ignore_index=True) # DataFrame hợp nhất
        # Hợp nhất tất cả DataFrame; True: đặt lại index

    def compare_models(self) -> pd.DataFrame:
        """So sánh các models"""
        all_results = self.load_all_results() # load model (CSV)

        if all_results.empty:
            return pd.DataFrame()

        # Pivot để so sánh
        comparison = all_results.pivot_table(
            index='model', # model
            columns='metric', # chỉ số thành cột
            values='value', # giá trị chỉ số
            aggfunc='last' # Lấy giá trị cuối cùng nếu có nhiều giá trị cho cùng một mô hình và chỉ số
        ).round(4)

        return comparison # DataFrame pivot