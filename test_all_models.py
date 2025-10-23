import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data.data_loader import MovieLensDataLoader
from src.utils.file_utils import load_model
from config.settings import config


def load_all_models():
    """Load all available models"""
    models_dir = Path(config.get('paths.models', 'saved_models')) # lấy path
    models = {}

    model_files = list(models_dir.glob('*.pkl')) # tìm file

    print(f"=====Loading Models from: {models_dir}")
    print("-" * 50)

    for model_file in model_files:
        try:
            model = load_model(model_file) # load model
            model_name = model_file.stem
            models[model_name] = model # name

            # model info
            model_type = model.__class__.__name__
            # para
            params = model.get_params()
            key_params = []

            if 'k' in params:
                key_params.append(f"k={params['k']}")
            if 'n_factors' in params:
                key_params.append(f"factors={params['n_factors']}")
            if 'embedding_dim' in params:
                key_params.append(f"embed={params['embedding_dim']}")
            if 'similarity' in params:
                key_params.append(f"sim={params['similarity']}")
            if 'based' in params:
                key_params.append(f"based={params['based']}")

            param_str = ", ".join(key_params) if key_params else "default"

            print(f"{model_name:<25} | {model_type:<20} | {param_str}")

        except Exception as e:
            print(f"{model_file.name:<25} | Failed to load: {e}")

    print("-" * 50)
    print(f"=====Total Models Loaded: {len(models)}")
    return models # dic {model_name: model_object}


def get_movie_info(movie_id: int, movies_df: pd.DataFrame):
    """Get detailed movie information"""
    if movie_id >= len(movies_df):
        return {
            'title': f'Movie {movie_id}',
            'main_genre': 'Unknown',
            'release_year': 'Unknown',
            'genres': []
        }

    # lấy thông tin phim từ movies_df theo id
    movie = movies_df.iloc[movie_id]
    return {
        'title': movie.get('title', f'Movie {movie_id}'),
        'main_genre': movie.get('main_genre', 'Unknown'),
        'release_year': movie.get('release_year', 'Unknown'),
        'genres': movie.get('genres', [])
    } # dictionary tt phim
# Hiển thị thông tin phim dự đoán, gợi ý


def predict_for_all_models(models: Dict, user_id: int, movie_id: int,
                           movies_df: pd.DataFrame) -> Dict[str, Any]:
    """Dự đoán điểm số tất cả mô hình cho 1 user-movie"""
    print(f"\n{'=' * 80}")
    print(f"=====PREDICTION COMPARISON - User {user_id}, Movie {movie_id}")
    print(f"{'=' * 80}")

    # Get movie info
    movie_info = get_movie_info(movie_id, movies_df)
    print(f"=====Movie: {movie_info['title']}")
    print(f"=====Genre: {movie_info['main_genre']}")
    print(f"=====Year: {movie_info['release_year']}")

    print(f"\n{'Model':<25} {'Type':<15} {'Rating':<8} {'Stars':<10} {'Time (ms)'}")
    print("-" * 80)

    predictions = {}

    for model_name, model in models.items():
        try:
            start_time = time.time()
            prediction = model.predict(user_id, movie_id)
            pred_time = (time.time() - start_time) * 1000  # Convert to ms

            stars = "★" * int(prediction) + "☆" * (5 - int(prediction))
            model_type = model.__class__.__name__[:14]  # Truncate long names

            print(f"{model_name:<25} {model_type:<15} {prediction:<8.2f} {stars:<10} {pred_time:<8.2f}")

            predictions[model_name] = {
                'rating': prediction,
                'time_ms': pred_time,
                'model_type': model.__class__.__name__
            } # save

        except Exception as e:
            print(f"{model_name:<25} {'ERROR':<15} {'N/A':<8} {'N/A':<10} {'N/A'}")
            predictions[model_name] = {'error': str(e)}

    print("-" * 80)

    # thống kê
    valid_predictions = [p['rating'] for p in predictions.values() if 'rating' in p]
    if valid_predictions:
        print(f"\n=====Prediction Statistics:")
        print(f"   Average: {np.mean(valid_predictions):.2f}")
        print(f"   Std Dev: {np.std(valid_predictions):.2f}")
        print(f"   Min: {np.min(valid_predictions):.2f}")
        print(f"   Max: {np.max(valid_predictions):.2f}")
        print(f"   Range: {np.max(valid_predictions) - np.min(valid_predictions):.2f}")

    return predictions # dic
# So sánh dự đoán giữa các mô hình, đánh giá thời gian thực thi


def recommend_for_all_models(models: Dict, user_id: int, movies_df: pd.DataFrame,
                             n_rec: int = 5) -> Dict[str, List]:
    """Tạo danh sách gợi ý từ tất cả mô hình cho một người dùng"""
    print(f"\n{'=' * 80}")
    print(f"=====RECOMMENDATION COMPARISON - User {user_id} (Top {n_rec})")
    print(f"{'=' * 80}")

    all_recommendations = {}

    for model_name, model in models.items():
        try:
            # đo time
            start_time = time.time()
            recommendations = model.get_user_recommendations(user_id, n_recommendations=n_rec)
            rec_time = (time.time() - start_time) * 1000

            print(f"\n====={model_name} ({model.__class__.__name__}) - {rec_time:.1f}ms")
            print(f"{'Rank':<4} {'Movie':<35} {'Genre':<15} {'Rating':<8} {'Stars'}")
            print("-" * 70)
            # danh sách
            for i, (movie_id, rating) in enumerate(recommendations, 1):
                movie_info = get_movie_info(movie_id, movies_df)
                title = movie_info['title'][:33] + ".." if len(movie_info['title']) > 35 else movie_info['title']
                genre = movie_info['main_genre'][:13] + ".." if len(movie_info['main_genre']) > 15 else movie_info[
                    'main_genre']
                stars = "★" * int(rating) + "☆" * (5 - int(rating))

                print(f"{i:<4} {title:<35} {genre:<15} {rating:<8.2f} {stars}")

            all_recommendations[model_name] = {
                'recommendations': recommendations,
                'time_ms': rec_time
            } # save

        except Exception as e:
            print(f"\n❌ {model_name}: Error - {e}")
            all_recommendations[model_name] = {'error': str(e)}

    return all_recommendations
# So sánh danh sách gợi ý giữa các mô hình, đánh giá thời gian thực thi

def analyze_model_agreement(predictions: Dict[str, Any]) -> None:
    """Phân tích độ đồng thuận giữa các mô hình dựa trên dự đoán điểm số"""
    print(f"\n{'=' * 80}")
    print("=====MODEL AGREEMENT ANALYSIS")
    print(f"{'=' * 80}")

    # lọc dự đoán có rating
    valid_models = [(name, data['rating']) for name, data in predictions.items()
                    if 'rating' in data]

    if len(valid_models) < 2: # < 2 model
        print("Need at least 2 models for agreement analysis")
        return

    print("Pairwise Rating Correlations:")
    print("-" * 40)

    model_names = [name for name, _ in valid_models]
    ratings_matrix = []

    # Tính trung bình điểm số
    print("=====Prediction Differences from Average:")
    ratings = [rating for _, rating in valid_models]
    avg_rating = np.mean(ratings)
    # hiển thị chênh lệch các mô hình với avg
    for name, rating in valid_models:
        diff = rating - avg_rating
        direction = "↗️" if diff > 0.1 else "↘️" if diff < -0.1 else "➡️"
        print(f"   {name:<25}: {diff:+.3f} {direction}")

    # tính std
    rating_std = np.std(ratings)
    # mức độ đồng thuận
    if rating_std < 0.2:
        agreement = "🟢 HIGH"
    elif rating_std < 0.5:
        agreement = "🟡 MEDIUM"
    else:
        agreement = "🔴 LOW"

    print(f"\nModel Agreement: {agreement} (std: {rating_std:.3f})")
# Đánh giá mức độ nhất quán giữa các mô hình


def performance_benchmark(models: Dict, user_ids: List[int], movie_ids: List[int]) -> None:
    """Đo hiệu suất dự đoán các mô hình trên nhiều cặp user-movie"""
    print(f"\n{'=' * 80}")
    print("=====PERFORMANCE BENCHMARK")
    print(f"{'=' * 80}")

    print(f"Testing {len(user_ids)} users × {len(movie_ids)} movies = {len(user_ids) * len(movie_ids)} predictions")
    print(f"\n{'Model':<25} {'Avg Time (ms)':<15} {'Total Time (s)':<15} {'Predictions/s'}")
    print("-" * 80)

    for model_name, model in models.items():
        try:
            times = []
            successful_predictions = 0

            start_total = time.time()

            # dự đoán cho từng cặp trong user_ids × movie_ids
            for user_id in user_ids:
                for movie_id in movie_ids:
                    try:
                        start_time = time.time()
                        model.predict(user_id, movie_id)
                        times.append((time.time() - start_time) * 1000)
                        successful_predictions += 1
                    except:
                        pass
            # time
            total_time = time.time() - start_total

            # số dự đoán/s
            if times:
                avg_time = np.mean(times)
                predictions_per_sec = successful_predictions / total_time

                print(f"{model_name:<25} {avg_time:<15.2f} {total_time:<15.2f} {predictions_per_sec:<15.1f}")
            else:
                print(f"{model_name:<25} {'ERROR':<15} {'ERROR':<15} {'ERROR'}")

        except Exception as e:
            print(f"{model_name:<25} {'ERROR':<15} {'ERROR':<15} {'ERROR'}")


def interactive_mode(models: Dict, data_loader: MovieLensDataLoader, movies_df: pd.DataFrame):
    """Interactive testing mode"""
    print(f"\n{'=' * 80}")
    print("INTERACTIVE MODEL TESTING")
    print("Commands:")
    print("  predict <user_id> <movie_id>     - Compare predictions")
    print("  recommend <user_id> [n]          - Compare recommendations")
    print("  benchmark [n_users] [n_movies]   - Performance benchmark")
    print("  agreement <user_id> <movie_id>   - Model agreement analysis")
    print("  info                             - Show dataset info")
    print("  models                           - List available models")
    print("  quit                             - Exit")
    print(f"{'=' * 80}")

    while True:
        try:
            command = input("\n>>> ").strip().split()

            if not command:
                continue

            cmd = command[0].lower()

            if cmd in ['quit', 'q', 'exit']:
                break

            elif cmd == 'predict' and len(command) >= 3:
                user_id = int(command[1])
                movie_id = int(command[2])
                predictions = predict_for_all_models(models, user_id, movie_id, movies_df)
                analyze_model_agreement(predictions)

            elif cmd == 'recommend' and len(command) >= 2:
                user_id = int(command[1])
                n_rec = int(command[2]) if len(command) > 2 else 5
                recommend_for_all_models(models, user_id, movies_df, n_rec)

            elif cmd == 'benchmark':
                n_users = int(command[1]) if len(command) > 1 else 10
                n_movies = int(command[2]) if len(command) > 2 else 10
                user_ids = np.random.choice(range(100), min(n_users, 100), replace=False)
                movie_ids = np.random.choice(range(200), min(n_movies, 200), replace=False)
                performance_benchmark(models, user_ids.tolist(), movie_ids.tolist())

            elif cmd == 'agreement' and len(command) >= 3:
                user_id = int(command[1])
                movie_id = int(command[2])
                predictions = predict_for_all_models(models, user_id, movie_id, movies_df)
                analyze_model_agreement(predictions)

            elif cmd == 'info':
                try:
                    stats = data_loader.get_stats()
                    print(f"\n=====Dataset Information:")
                    print(f"  Users: {stats['dataset']['n_users']:,}")
                    print(f"  Movies: {stats['dataset']['n_movies']:,}")
                    print(f"  Ratings: {stats['dataset']['n_ratings_train']:,}")
                    print(f"  Sparsity: {stats['dataset']['sparsity']:.1%}")
                except Exception as e:
                    print(f"Error getting stats: {e}")

            elif cmd == 'models':
                print(f"\n=====Available Models:")
                for i, (model_name, model) in enumerate(models.items(), 1):
                    model_type = model.__class__.__name__
                    print(f"  {i}. {model_name} ({model_type})")

            else:
                print("Invalid command. Type 'quit' to exit.")

        except (ValueError, IndexError) as e:
            print(f"Invalid command format: {e}")
        except KeyboardInterrupt:
            print(f"\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test All Recommendation Models')
    parser.add_argument('--user-id', type=int, help='User ID for testing')
    parser.add_argument('--movie-id', type=int, help='Movie ID for prediction comparison')
    parser.add_argument('--recommend', action='store_true', help='Get recommendations')
    parser.add_argument('--n-recommendations', type=int, default=5, help='Number of recommendations')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')

    args = parser.parse_args()

    print("=====Model Tester")
    print("=" * 60)

    try:
        print("\n=====Loading data...")
        data_path = config.get('data.raw_path')
        data_loader = MovieLensDataLoader(data_path)
        ratings_train, ratings_test = data_loader.load_ratings()
        movies_df = data_loader.load_movies()
        print(f"Loaded {len(movies_df)} movies, {len(ratings_train)} ratings")

        print("\n=====Loading models...")
        models = load_all_models()

        if not models:
            print("❌ No models found!")
            return

        print(f"Ready to test with {len(models)} models")

        # thực thi trên tham số
        if args.interactive:
            interactive_mode(models, data_loader, movies_df)

        elif args.user_id is not None:
            if args.movie_id is not None:
                # Single prediction comparison
                predictions = predict_for_all_models(models, args.user_id, args.movie_id, movies_df)
                analyze_model_agreement(predictions)
            elif args.recommend:
                # Recommendation comparison
                recommend_for_all_models(models, args.user_id, movies_df, args.n_recommendations)
            else:
                # prediction + recommendation
                sample_movie = np.random.randint(0, len(movies_df))
                predictions = predict_for_all_models(models, args.user_id, sample_movie, movies_df)
                recommend_for_all_models(models, args.user_id, movies_df, args.n_recommendations)

        elif args.benchmark:
            # Performance benchmark
            user_ids = np.random.choice(range(100), 20, replace=False).tolist()
            movie_ids = np.random.choice(range(200), 20, replace=False).tolist()
            performance_benchmark(models, user_ids, movie_ids)

        else:
            parser.print_help()
            interactive_mode(models, data_loader, movies_df)

    except KeyboardInterrupt:
        print(f"\n\n=====Goodbye!")
    except Exception as e:
        print(f"=====Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()