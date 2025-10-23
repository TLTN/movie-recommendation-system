import numpy as np
from typing import List


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def precision_at_k(recommendations: List[List[int]], relevant_items: List[List[int]], k: int) -> float:
    precisions = []
    # tỷ lệ các mục được gợi ý trong top-K thực sự phù hợp với danh sách các mục liên quan của người dùng.

    for user_recs, user_relevant in zip(recommendations, relevant_items):
        if not user_recs or not user_relevant: # == null
            continue

        # lấy top K gợi ý, chuyển thành tập
        top_k_recs = set(user_recs[:k])
        relevant_set = set(user_relevant)

        # Tính số mục đúng
        hits = len(top_k_recs & relevant_set) # giao tập
        precision = hits / min(k, len(user_recs)) if user_recs else 0.0
        precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def recall_at_k(recommendations: List[List[int]], relevant_items: List[List[int]], k: int) -> float:
    recalls = []
    # tỷ lệ các mục liên quan được bao phủ bởi top-K gợi ý của người dùng
    # recommendations: danh sách gợi ý
    # relevant_items: danh sách mục liên quan
    for user_recs, user_relevant in zip(recommendations, relevant_items):
        if not user_relevant:
            continue

        # lấy top K gợi ý, chuyển thành tập
        top_k_recs = set(user_recs[:k])
        relevant_set = set(user_relevant)

        # Tính recall
        hits = len(top_k_recs & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 0.0
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def ndcg_at_k(recommendations: List[List[int]], relevant_items: List[List[int]], k: int = 10) -> float:
    ndcg_scores = []
    # ưu tiên các mục liên quan xuất hiện sớm trong danh sách top-K

    for user_recs, user_relevant in zip(recommendations, relevant_items):
        if not user_recs or not user_relevant:
            continue

        # Tạo bản đồ độ liên quan: 1 - liên quan
        relevance_map = {item: 1.0 for item in user_relevant}

        # Tính DCG
        dcg = 0.0
        for j, item in enumerate(user_recs[:k]):
            if item in relevance_map:
                relevance = relevance_map[item]
                dcg += relevance / np.log2(j + 2)  # j+2, vì log2(1) = 0

        # Tính IDCG (ideal DCG)
        ideal_relevances = [1.0] * min(len(user_relevant), k) # danh sách lý tưởng, 1.0 cho số lượng mục liên quan
        idcg = sum(rel / np.log2(j + 2) for j, rel in enumerate(ideal_relevances))

        # Tính NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0.0
    # Đánh giá chất lượng xếp hạng của gợi ý, ưu tiên các mục liên quan ở vị trí cao


def hit_rate_at_k(recommendations: List[List[int]], relevant_items: List[List[int]], k: int) -> float:
    """tỷ lệ người dùng có ít nhất một mục liên quan trong top-K"""
    hits = 0
    total_users = 0

    for user_recs, user_relevant in zip(recommendations, relevant_items):
        if not user_relevant:
            continue

        total_users += 1

        # Check if any of top-k recommendations are relevant
        top_k_recs = set(user_recs[:k]) # Lấy top-K
        relevant_set = set(user_relevant) # chuyển thành tập

        if len(top_k_recs & relevant_set) > 0: # 1 đúng, != null
            hits += 1

    return hits / total_users if total_users > 0 else 0.0
    # Đo khả năng mô hình đưa ra ít nhất một gợi ý đúng cho người dùng