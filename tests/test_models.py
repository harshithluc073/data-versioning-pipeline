"""
Unit Tests for Model Training and Evaluation
Tests for model training, evaluation, and predictions
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class TestModelTraining:
    """Test suite for model training"""
    
    @pytest.fixture
    def sample_training_data(self, tmp_path):
        """Create sample training data"""
        train_data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50),
            'feature4': np.random.randn(50),
            'feature5': np.random.randn(50),
            'feature_ratio': np.random.randn(50),
            'feature_sum': np.random.randn(50),
            'target': np.random.choice([0, 1, 2], 50)
        })
        
        train_path = tmp_path / "processed" / "train.csv"
        os.makedirs(train_path.parent, exist_ok=True)
        train_data.to_csv(train_path, index=False)
        
        return str(train_path)
    
    @pytest.fixture
    def trained_model(self, sample_training_data):
        """Train a simple model for testing"""
        # Load data
        data = pd.read_csv(sample_training_data)
        X = data.drop(columns=['target'])
        y = data['target']
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    def test_model_can_fit(self, sample_training_data):
        """Test that model can be trained"""
        data = pd.read_csv(sample_training_data)
        X = data.drop(columns=['target'])
        y = data['target']
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_model_makes_predictions(self, trained_model, sample_training_data):
        """Test that trained model can make predictions"""
        data = pd.read_csv(sample_training_data)
        X = data.drop(columns=['target'])
        
        predictions = trained_model.predict(X)
        
        assert predictions is not None
        assert len(predictions) == len(X)
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_model_has_feature_importances(self, trained_model):
        """Test that model provides feature importances"""
        assert hasattr(trained_model, 'feature_importances_')
        assert len(trained_model.feature_importances_) == 7  # 7 features
        assert all(imp >= 0 for imp in trained_model.feature_importances_)
        assert abs(sum(trained_model.feature_importances_) - 1.0) < 0.01
    
    def test_model_predict_proba(self, trained_model, sample_training_data):
        """Test that model provides probability predictions"""
        data = pd.read_csv(sample_training_data)
        X = data.drop(columns=['target'])
        
        probas = trained_model.predict_proba(X)
        
        assert probas is not None
        assert probas.shape == (len(X), 3)  # 3 classes
        assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_model_score(self, trained_model, sample_training_data):
        """Test that model can compute accuracy score"""
        data = pd.read_csv(sample_training_data)
        X = data.drop(columns=['target'])
        y = data['target']
        
        score = trained_model.score(X, y)
        
        assert score is not None
        assert 0.0 <= score <= 1.0
    
    def test_model_can_be_saved_loaded(self, trained_model, tmp_path):
        """Test model serialization"""
        model_path = tmp_path / "test_model.pkl"
        
        # Save model
        joblib.dump(trained_model, model_path)
        assert model_path.exists()
        
        # Load model
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None
        assert type(loaded_model) == type(trained_model)


class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    @pytest.fixture
    def sample_test_data(self, tmp_path):
        """Create sample test data"""
        test_data = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20),
            'feature4': np.random.randn(20),
            'feature5': np.random.randn(20),
            'feature_ratio': np.random.randn(20),
            'feature_sum': np.random.randn(20),
            'target': np.random.choice([0, 1, 2], 20)
        })
        
        test_path = tmp_path / "processed" / "test.csv"
        os.makedirs(test_path.parent, exist_ok=True)
        test_data.to_csv(test_path, index=False)
        
        return str(test_path), test_data
    
    @pytest.fixture
    def model_and_predictions(self, sample_test_data):
        """Train model and make predictions"""
        test_path, test_data = sample_test_data
        
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)  # Overfitting for testing
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        return model, y_test, y_pred
    
    def test_accuracy_calculation(self, model_and_predictions):
        """Test accuracy metric calculation"""
        from sklearn.metrics import accuracy_score
        
        model, y_test, y_pred = model_and_predictions
        accuracy = accuracy_score(y_test, y_pred)
        
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0
    
    def test_precision_calculation(self, model_and_predictions):
        """Test precision metric calculation"""
        from sklearn.metrics import precision_score
        
        model, y_test, y_pred = model_and_predictions
        precision = precision_score(y_test, y_pred, average='weighted')
        
        assert precision is not None
        assert 0.0 <= precision <= 1.0
    
    def test_recall_calculation(self, model_and_predictions):
        """Test recall metric calculation"""
        from sklearn.metrics import recall_score
        
        model, y_test, y_pred = model_and_predictions
        recall = recall_score(y_test, y_pred, average='weighted')
        
        assert recall is not None
        assert 0.0 <= recall <= 1.0
    
    def test_f1_score_calculation(self, model_and_predictions):
        """Test F1 score calculation"""
        from sklearn.metrics import f1_score
        
        model, y_test, y_pred = model_and_predictions
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        assert f1 is not None
        assert 0.0 <= f1 <= 1.0
    
    def test_confusion_matrix_generation(self, model_and_predictions):
        """Test confusion matrix generation"""
        from sklearn.metrics import confusion_matrix
        
        model, y_test, y_pred = model_and_predictions
        cm = confusion_matrix(y_test, y_pred)
        
        assert cm is not None
        assert cm.shape == (3, 3)  # 3 classes
        assert cm.sum() == len(y_test)
    
    def test_classification_report(self, model_and_predictions):
        """Test classification report generation"""
        from sklearn.metrics import classification_report
        
        model, y_test, y_pred = model_and_predictions
        report = classification_report(y_test, y_pred, output_dict=True)
        
        assert report is not None
        assert 'accuracy' in report
        assert '0' in report  # Class 0
        assert '1' in report  # Class 1
        assert '2' in report  # Class 2


class TestPredictionPipeline:
    """Test suite for end-to-end prediction pipeline"""
    
    @pytest.fixture
    def prediction_setup(self, tmp_path):
        """Setup for prediction tests"""
        # Create sample data
        X_train = pd.DataFrame({
            'feature1': np.random.randn(30),
            'feature2': np.random.randn(30),
            'feature3': np.random.randn(30),
            'feature4': np.random.randn(30),
            'feature5': np.random.randn(30),
            'feature_ratio': np.random.randn(30),
            'feature_sum': np.random.randn(30),
        })
        y_train = np.random.choice([0, 1, 2], 30)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)
        
        return str(model_path), X_train.iloc[0].to_dict()
    
    def test_load_model_and_predict(self, prediction_setup):
        """Test loading model and making prediction"""
        model_path, sample_input = prediction_setup
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare input
        X = pd.DataFrame([sample_input])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        assert prediction is not None
        assert prediction in [0, 1, 2]
    
    def test_batch_prediction(self, prediction_setup):
        """Test batch prediction"""
        model_path, sample_input = prediction_setup
        
        # Load model
        model = joblib.load(model_path)
        
        # Create batch input
        batch_data = pd.DataFrame([sample_input] * 5)
        
        # Make predictions
        predictions = model.predict(batch_data)
        
        assert len(predictions) == 5
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_prediction_with_probabilities(self, prediction_setup):
        """Test prediction with confidence scores"""
        model_path, sample_input = prediction_setup
        
        # Load model
        model = joblib.load(model_path)
        
        # Prepare input
        X = pd.DataFrame([sample_input])
        
        # Get probabilities
        probas = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        confidence = probas[prediction]
        
        assert 0.0 <= confidence <= 1.0
        assert confidence == max(probas)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])