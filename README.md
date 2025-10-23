# Movie Recommendation System

Hệ thống đề xuất phim sử dụng thuật toán Collaborative Filtering với dataset MovieLens 100K.

## 📋 Tổng quan

Dự án này xây dựng một hệ thống đề xuất phim hoàn chỉnh với các tính năng:

- **Thuật toán đa dạng**: KNN, Matrix Factorization, Neural Collaborative Filtering
- **Đánh giá toàn diện**: RMSE, MAE, Precision@K, Recall@K, NDCG@K
- **Giao diện web**: Flask web application với template đẹp
- **Phân tích dữ liệu**: Jupyter notebooks với visualization
- **Cấu trúc mở rộng**: Dễ dàng thêm model mới

## 🚀 Cài đặt

### 1. Clone repository
```bash
git https://github.com/TLTN/movie-recommendation-system
cd BTL
```

### 2. Tạo virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Download MovieLens 100K dataset
```bash
mkdir -p data/raw
cd data/raw
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ../..
```

## 📊 Dataset

**MovieLens 100K** bao gồm:
- 100,000 ratings (1-5 stars)
- 943 users
- 1,682 movies
- Thông tin demographics của users
- Metadata của movies (genre, release date, etc.)

## 🔧 Sử dụng

### 1. Training Models

```bash
# Train KNN model
python experiments/train_knn.py --k 40 --similarity cosine --based user

# Train với tham số khác
python experiments/train_knn.py --k 30 --similarity pearson --based item
```

### 2. Chạy Web Application

```bash
python src/web/app.py
```

Truy cập: http://localhost:5000

### 3. Sử dụng API

```python
import requests

# Get recommendations
response = requests.get('http://localhost:5000/api/recommend', 
                       params={'user_id': 1, 'n_recommendations': 10})
recommendations = response.json()

# Train new model
train_data = {
    'model_type': 'knn',
    'parameters': {'k': 40, 'similarity': 'cosine', 'based': 'user'}
}
response = requests.post('http://localhost:5000/api/train', json=train_data)
```

## 🤖 Models

### 1. K-Nearest Neighbors (KNN)
- **User-based**: Tìm users tương tự
- **Item-based**: Tìm items tương tự
- **Similarity**: Cosine, Pearson

### 2. Matrix Factorization (SVD)
- Phân tích ma trận user-item
- Latent factors
- Regularization

### 3. Neural Collaborative Filtering
- Deep learning approach
- Embedding layers
- Multi-layer perceptron

## 📈 Evaluation Metrics

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **Precision@K**: Precision at K
- **Recall@K**: Recall at K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate**: Tỷ lệ hit

## 🌐 Web Interface

- **Trang chủ**: Overview và navigation
- **Recommendations**: Đề xuất phim cho user
- **Training**: Train model mới
- **API**: RESTful API endpoints

## 🔬 Experiments

### Chạy experiment:
```bash
python experiments/train_knn.py --k 40 --similarity cosine --evaluate
```

### Tham số có thể điều chỉnh:
- `--k`: Số lượng neighbors (default: 40)
- `--similarity`: Cosine hoặc Pearson (default: cosine)  
- `--based`: User-based hoặc Item-based (default: user)
- `--evaluate`: Đánh giá model sau khi train

## 🧪 Testing

```bash
# Run tests (khi có)
pytest tests/
```

## 📝 Configuration

Chỉnh sửa `config/config.yaml` để:
- Thay đổi hyperparameters
- Cấu hình evaluation metrics
- Điều chỉnh paths
- Setup web application

## 🚀 Mở rộng

### Thêm metric mới:
1. Implement trong `src/evaluation/metrics.py`
2. Update `Evaluator` class
3. Thêm vào config

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📄 License

MIT License

## 👨‍💻 Author

**TLTN**
- Email: tltnlovelala@example.com
- GitHub: [@TLTN](https://github.com/TLTN)

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) dataset
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- [Flask](https://flask.palletsprojects.com/) for web framework