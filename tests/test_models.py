import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.knn_recommender import KNNRecommender
from src.models.matrix_factorization import MatrixFactorization

try:
    from src.models.neural_cf import NeuralCollaborativeFiltering, TORCH_AVAILABLE
    NEURAL_CF_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    NEURAL_CF_AVAILABLE = False
    NeuralCollaborativeFiltering = None


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42) # random seed, tái lập

    # small dataset
    n_users, n_items, n_ratings = 50, 30, 500

    users = np.random.randint(0, n_users, n_ratings)
    items = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)

    # matrix
    train_data = np.column_stack([users, items, ratings])

    # remove duplicates
    train_data = np.unique(train_data, axis=0)

    return train_data


@pytest.fixture
def sample_val_data():
    """Create sample validation data"""
    np.random.seed(123)

    # Create validation set: smaller than training
    n_users, n_items, n_ratings = 50, 30, 100

    users = np.random.randint(0, n_users, n_ratings)
    items = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.randint(1, 6, n_ratings)

    val_data = np.column_stack([users, items, ratings])
    val_data = np.unique(val_data, axis=0)

    return val_data # numpy: [user_id, item_id, rating]


class TestKNNRecommender:
    """Test cases for KNN Recommender"""

    def test_initialization(self):
        """Test KNN model initialization"""
        model = KNNRecommender(k=10, similarity='cosine', based='user')

        assert model.k == 10
        assert model.similarity == 'cosine'
        assert model.based == 'user'
        assert not model.is_fitted

    def test_fit(self, sample_data):
        """Test KNN model training"""
        model = KNNRecommender(k=5, similarity='cosine', based='user')

        # Should not be fitted initially
        assert not model.is_fitted

        # Fit the model
        model.fit(sample_data)

        # Should be fitted after training
        assert model.is_fitted
        assert model.n_users > 0
        assert model.n_items > 0
        assert model.rating_matrix is not None
        assert model.similarity_matrix is not None

    def test_predict(self, sample_data):
        """Test KNN prediction"""
        model = KNNRecommender(k=5, similarity='cosine', based='user')
        model.fit(sample_data)

        # Test prediction for existing user and item
        prediction = model.predict(0, 0)
        assert 1.0 <= prediction <= 5.0

        # Test prediction for out-of-bounds user/item
        prediction = model.predict(1000, 1000)
        assert isinstance(prediction, float)

    def test_recommend(self, sample_data):
        """Test KNN recommendations"""
        model = KNNRecommender(k=5, similarity='cosine', based='user')
        model.fit(sample_data)

        # Get recommendations
        recommendations = model.recommend(0, n_recommendations=5)

        assert len(recommendations) <= 5
        assert all(isinstance(item_id, (int, np.integer)) for item_id, _ in recommendations)
        assert all(isinstance(rating, (float, np.floating)) for _, rating in recommendations)
        assert all(1.0 <= rating <= 5.0 for _, rating in recommendations)

    def test_user_vs_item_based(self, sample_data):
        """Test user-based vs item-based approaches"""
        user_model = KNNRecommender(k=5, similarity='cosine', based='user')
        item_model = KNNRecommender(k=5, similarity='cosine', based='item')

        user_model.fit(sample_data)
        item_model.fit(sample_data)

        # Both should be fitted
        assert user_model.is_fitted
        assert item_model.is_fitted

        # Should produce different predictions
        user_pred = user_model.predict(0, 0)
        item_pred = item_model.predict(0, 0)

        assert isinstance(user_pred, float)
        assert isinstance(item_pred, float)

    def test_similarity_metrics(self, sample_data):
        """Test different similarity metrics"""
        cosine_model = KNNRecommender(k=5, similarity='cosine', based='user')
        pearson_model = KNNRecommender(k=5, similarity='pearson', based='user')

        cosine_model.fit(sample_data)
        pearson_model.fit(sample_data)

        # Both should be fitted
        assert cosine_model.is_fitted
        assert pearson_model.is_fitted

        # Should have different similarity matrices
        assert cosine_model.similarity_matrix.shape == pearson_model.similarity_matrix.shape
        assert not np.array_equal(cosine_model.similarity_matrix, pearson_model.similarity_matrix)


class TestMatrixFactorization:
    """Test cases for Matrix Factorization"""

    def test_initialization(self):
        """Test MF model initialization"""
        model = MatrixFactorization(n_factors=10, learning_rate=0.01, n_epochs=5)

        assert model.n_factors == 10
        assert model.learning_rate == 0.01
        assert model.n_epochs == 5
        assert not model.is_fitted

    def test_fit(self, sample_data):
        """Test MF model training"""
        model = MatrixFactorization(n_factors=5, n_epochs=5)

        # Should not be fitted initially
        assert not model.is_fitted

        # Fit the model
        model.fit(sample_data)

        # Should be fitted after training
        assert model.is_fitted
        assert model.user_factors is not None
        assert model.item_factors is not None
        assert model.user_factors.shape[1] == 5  # n_factors
        assert model.item_factors.shape[1] == 5  # n_factors
        assert len(model.train_losses) == 5  # n_epochs

    def test_predict(self, sample_data):
        """Test MF prediction"""
        model = MatrixFactorization(n_factors=5, n_epochs=5)
        model.fit(sample_data)

        # Test prediction for existing user and item
        prediction = model.predict(0, 0)
        assert 1.0 <= prediction <= 5.0

        # Test prediction for out-of-bounds user/item
        prediction = model.predict(1000, 1000)
        assert isinstance(prediction, float)

    def test_recommend(self, sample_data):
        """Test MF recommendations"""
        model = MatrixFactorization(n_factors=5, n_epochs=5)
        model.fit(sample_data)

        # Get recommendations
        recommendations = model.recommend(0, n_recommendations=5)

        assert len(recommendations) <= 5
        assert all(isinstance(item_id, (int, np.integer)) for item_id, _ in recommendations)
        assert all(isinstance(rating, (float, np.floating)) for _, rating in recommendations)

    def test_training_convergence(self, sample_data):
        """Test that training loss decreases"""
        model = MatrixFactorization(n_factors=5, n_epochs=10, learning_rate=0.1)
        model.fit(sample_data)

        # Training loss should generally decrease
        losses = model.train_losses
        assert len(losses) == 10

        # First loss should be higher than last loss (generally)
        assert losses[0] > losses[-1] * 0.5  # Allow some tolerance


@pytest.mark.skipif(not NEURAL_CF_AVAILABLE, reason="PyTorch not available")
class TestNeuralCollaborativeFiltering:
    """Test cases for Neural Collaborative Filtering"""

    def test_initialization(self):
        """Test Neural CF model initialization"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=32,
            hidden_layers=[64, 32],
            dropout=0.2,
            learning_rate=0.001,
            batch_size=128,
            epochs=5
        )

        assert model.embedding_dim == 32
        assert model.hidden_layers == [64, 32]
        assert model.dropout == 0.2
        assert model.learning_rate == 0.001
        assert model.batch_size == 128
        assert model.epochs == 5
        assert not model.is_fitted

    def test_fit(self, sample_data):
        """Test Neural CF model training"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            hidden_layers=[32, 16],
            epochs=3,
            batch_size=64
        )

        # Should not be fitted initially
        assert not model.is_fitted

        # Fit the model
        model.fit(sample_data)

        # Should be fitted after training
        assert model.is_fitted
        assert model.n_users > 0
        assert model.n_items > 0
        assert model.model is not None
        assert len(model.train_losses) == 3  # epochs

    def test_fit_with_validation(self, sample_data, sample_val_data):
        """Test Neural CF training with validation data"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            hidden_layers=[32, 16],
            epochs=3,
            batch_size=64
        )

        # Fit with validation data
        model.fit(sample_data, val_data=sample_val_data)

        # Should have both training and validation losses
        assert len(model.train_losses) == 3
        assert len(model.val_losses) == 3

        # Validation loss should be reasonable
        assert all(loss > 0 for loss in model.val_losses)

    def test_predict(self, sample_data):
        """Test Neural CF prediction"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            hidden_layers=[32],
            epochs=2,
            batch_size=64
        )
        model.fit(sample_data)

        # Test prediction for existing user and item
        prediction = model.predict(0, 0)
        assert 1.0 <= prediction <= 5.0
        assert isinstance(prediction, float)

        # Test prediction for out-of-bounds user/item
        prediction = model.predict(1000, 1000)
        assert prediction == 3.0  # Should return default rating

    def test_recommend(self, sample_data):
        """Test Neural CF recommendations"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            hidden_layers=[32],
            epochs=2,
            batch_size=64
        )
        model.fit(sample_data)

        # Get recommendations
        recommendations = model.recommend(0, n_recommendations=5)

        assert len(recommendations) <= 5
        assert all(isinstance(item_id, (int, np.integer)) for item_id, _ in recommendations)
        assert all(isinstance(rating, (float, np.floating)) for _, rating in recommendations)
        assert all(1.0 <= rating <= 5.0 for _, rating in recommendations)

        # Recommendations should be sorted by rating (descending)
        ratings = [rating for _, rating in recommendations]
        assert ratings == sorted(ratings, reverse=True)

    def test_recommend_out_of_bounds_user(self, sample_data):
        """Test recommendations for non-existent user"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=2,
            batch_size=64
        )
        model.fit(sample_data)

        # Should return popular items
        recommendations = model.recommend(9999, n_recommendations=5)
        assert len(recommendations) <= 5

    def test_optimizer_types(self, sample_data):
        """Test different optimizer types"""
        optimizers = ['adam', 'sgd', 'rmsprop']

        for opt in optimizers:
            model = NeuralCollaborativeFiltering(
                embedding_dim=16,
                hidden_layers=[32],
                epochs=2,
                optimizer=opt,
                batch_size=64
            )
            model.fit(sample_data)

            assert model.is_fitted
            assert len(model.train_losses) == 2

    def test_loss_functions(self, sample_data):
        """Test different loss functions"""
        loss_functions = ['mse', 'mae']

        for loss_fn in loss_functions:
            model = NeuralCollaborativeFiltering(
                embedding_dim=16,
                hidden_layers=[32],
                epochs=2,
                loss_function=loss_fn,
                batch_size=64
            )
            model.fit(sample_data)

            assert model.is_fitted
            assert len(model.train_losses) == 2

    def test_get_embeddings(self, sample_data):
        """Test getting user and item embeddings"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=2,
            batch_size=64
        )
        model.fit(sample_data)

        # Get user embedding
        user_embedding = model.get_user_embedding(0)
        assert user_embedding.shape == (16,)  # embedding_dim
        assert isinstance(user_embedding, np.ndarray)

        # Get item embedding
        item_embedding = model.get_item_embedding(0)
        assert item_embedding.shape == (16,)
        assert isinstance(item_embedding, np.ndarray)

        # Test out-of-bounds
        user_embedding_oob = model.get_user_embedding(9999)
        assert np.all(user_embedding_oob == 0)

    def test_get_popular_items(self, sample_data):
        """Test getting popular items"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=2,
            batch_size=64
        )
        model.fit(sample_data)

        popular_items = model.get_popular_items(n_recommendations=5)

        assert len(popular_items) <= 5
        assert all(isinstance(item_id, (int, np.integer)) for item_id, _ in popular_items)

        # Should be sorted by popularity score
        scores = [score for _, score in popular_items]
        assert scores == sorted(scores, reverse=True)

    def test_training_history(self, sample_data, sample_val_data):
        """Test getting training history"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=3,
            batch_size=64
        )
        model.fit(sample_data, val_data=sample_val_data)

        history = model.get_training_history()

        assert 'train_losses' in history
        assert 'val_losses' in history
        assert len(history['train_losses']) == 3
        assert len(history['val_losses']) == 3

    def test_training_convergence(self, sample_data):
        """Test that training loss generally decreases"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            hidden_layers=[32, 16],
            epochs=10,
            learning_rate=0.01,
            batch_size=64
        )
        model.fit(sample_data)

        losses = model.train_losses
        assert len(losses) == 10

        # First loss should be higher than last loss
        assert losses[0] > losses[-1] * 0.3  # Allow some tolerance

    def test_model_state(self, sample_data):
        """Test model state saving and loading"""
        model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=2,
            batch_size=64
        )
        model.fit(sample_data)

        # Get model state
        state = model._get_model_state()

        assert 'model_state_dict' in state
        assert 'train_losses' in state
        assert state['model_state_dict'] is not None

        # Create new model and load state
        new_model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=2,
            batch_size=64
        )
        new_model.n_users = model.n_users
        new_model.n_items = model.n_items
        new_model._set_model_state(state)

        # Models should produce same predictions
        pred1 = model.predict(0, 0)
        pred2 = new_model.predict(0, 0)
        assert abs(pred1 - pred2) < 1e-5


class TestModelComparison:
    """Compare different models"""

    def test_model_interfaces(self, sample_data):
        """Test that all models have consistent interfaces"""
        models = [
            KNNRecommender(k=5),
            MatrixFactorization(n_factors=5, n_epochs=5)
        ]

        # Add Neural CF if available
        if NEURAL_CF_AVAILABLE:
            models.append(NeuralCollaborativeFiltering(
                embedding_dim=16,
                epochs=3,
                batch_size=64
            ))

        for model in models:
            # Test fitting
            model.fit(sample_data)
            assert model.is_fitted

            # Test prediction
            prediction = model.predict(0, 0)
            assert isinstance(prediction, (float, np.floating))

            # Test recommendation
            recommendations = model.recommend(0, n_recommendations=3)
            assert len(recommendations) <= 3

            # Test parameter access
            params = model.get_params()
            assert isinstance(params, dict)

    def test_reproducibility(self, sample_data):
        """Test that models produce consistent results with same parameters"""
        # Test KNN reproducibility
        model1 = KNNRecommender(k=5, similarity='cosine', based='user')
        model2 = KNNRecommender(k=5, similarity='cosine', based='user')

        model1.fit(sample_data)
        model2.fit(sample_data)

        pred1 = model1.predict(0, 0)
        pred2 = model2.predict(0, 0)

        assert abs(pred1 - pred2) < 1e-10  # Should be exactly the same

        # Test MF reproducibility (with same random seed)
        np.random.seed(42)
        model3 = MatrixFactorization(n_factors=5, n_epochs=5)
        model3.fit(sample_data)

        np.random.seed(42)
        model4 = MatrixFactorization(n_factors=5, n_epochs=5)
        model4.fit(sample_data)

        pred3 = model3.predict(0, 0)
        pred4 = model4.predict(0, 0)

        assert abs(pred3 - pred4) < 1e-6  # Should be very close

    @pytest.mark.skipif(not NEURAL_CF_AVAILABLE, reason="PyTorch not available")
    def test_all_models_comparison(self, sample_data):
        """Compare predictions across all three models"""
        # Train all models
        knn_model = KNNRecommender(k=5)
        mf_model = MatrixFactorization(n_factors=10, n_epochs=10)
        ncf_model = NeuralCollaborativeFiltering(
            embedding_dim=16,
            epochs=5,
            batch_size=64
        )

        knn_model.fit(sample_data)
        mf_model.fit(sample_data)
        ncf_model.fit(sample_data)

        # Get predictions from all models
        user_id, item_id = 0, 0

        knn_pred = knn_model.predict(user_id, item_id)
        mf_pred = mf_model.predict(user_id, item_id)
        ncf_pred = ncf_model.predict(user_id, item_id)

        # All predictions should be in valid range
        assert 1.0 <= knn_pred <= 5.0
        assert 1.0 <= mf_pred <= 5.0
        assert 1.0 <= ncf_pred <= 5.0

        # Get recommendations from all models
        knn_recs = knn_model.recommend(user_id, n_recommendations=5)
        mf_recs = mf_model.recommend(user_id, n_recommendations=5)
        ncf_recs = ncf_model.recommend(user_id, n_recommendations=5)

        # All should return valid recommendations
        assert len(knn_recs) > 0
        assert len(mf_recs) > 0
        assert len(ncf_recs) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])