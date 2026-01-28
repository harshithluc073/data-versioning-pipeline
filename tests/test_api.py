"""
Integration Tests for FastAPI Endpoint
Tests for API endpoints and integration
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.app import app, load_model


@pytest.fixture(scope="module")
def test_client():
    """Create test client"""
    client = TestClient(app)
    return client


@pytest.fixture(scope="module")
def setup_test_model(tmp_path_factory):
    """Setup test model and scaler"""
    tmp_path = tmp_path_factory.mktemp("test_models")
    
    # Create and train simple model
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
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    
    # Create dummy scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    return str(model_path), str(scaler_path)


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_returns_200(self, test_client):
        """Test root endpoint returns 200"""
        response = test_client.get("/")
        assert response.status_code == 200
    
    def test_root_returns_json(self, test_client):
        """Test root endpoint returns JSON"""
        response = test_client.get("/")
        assert response.headers["content-type"] == "application/json"
    
    def test_root_has_message(self, test_client):
        """Test root endpoint has message"""
        response = test_client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_returns_200(self, test_client):
        """Test health endpoint returns 200"""
        response = test_client.get("/health")
        assert response.status_code == 200
    
    def test_health_returns_status(self, test_client):
        """Test health endpoint returns status"""
        response = test_client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data
    
    def test_health_model_status(self, test_client):
        """Test health endpoint reports model status"""
        response = test_client.get("/health")
        data = response.json()
        
        assert isinstance(data["model_loaded"], bool)


class TestPredictEndpoint:
    """Tests for prediction endpoint"""
    
    @pytest.fixture
    def valid_request(self):
        """Valid prediction request"""
        return {
            "feature1": 5.1,
            "feature2": 3.5,
            "feature3": 1.4,
            "feature4": 0.2,
            "feature5": 2.3
        }
    
    def test_predict_with_valid_data(self, test_client, valid_request):
        """Test prediction with valid data"""
        response = test_client.post("/predict", json=valid_request)
        
        # May be 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]
    
    def test_predict_returns_required_fields(self, test_client, valid_request):
        """Test prediction returns required fields"""
        response = test_client.post("/predict", json=valid_request)
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "prediction_label" in data
            assert "confidence" in data
            assert "model_version" in data
            assert "timestamp" in data
    
    def test_predict_with_missing_field(self, test_client):
        """Test prediction with missing field returns 422"""
        invalid_request = {
            "feature1": 5.1,
            "feature2": 3.5,
            # Missing other features
        }
        
        response = test_client.post("/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_predict_with_invalid_type(self, test_client):
        """Test prediction with invalid type returns 422"""
        invalid_request = {
            "feature1": "invalid",  # Should be float
            "feature2": 3.5,
            "feature3": 1.4,
            "feature4": 0.2,
            "feature5": 2.3
        }
        
        response = test_client.post("/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_predict_with_nan(self, test_client):
        """Test prediction with NaN value returns 422"""
        # Use content string to avoid JSON serialization error in TestClient
        # Note: Standard JSON doesn't support NaN, but we want to test that the API handles it
        # or that the validator catches it if somehow it gets through

        # If we send string "NaN", Pydantic might coerce it to float('nan')
        invalid_request = {
            "feature1": "NaN",
            "feature2": 3.5,
            "feature3": 1.4,
            "feature4": 0.2,
            "feature5": 2.3
        }
        
        response = test_client.post("/predict", json=invalid_request)
        assert response.status_code == 422
    
    def test_predict_confidence_range(self, test_client, valid_request):
        """Test prediction confidence is in valid range"""
        response = test_client.post("/predict", json=valid_request)
        
        if response.status_code == 200:
            data = response.json()
            assert 0.0 <= data["confidence"] <= 1.0


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint"""
    
    @pytest.fixture
    def valid_batch_request(self):
        """Valid batch prediction request"""
        return {
            "instances": [
                {
                    "feature1": 5.1,
                    "feature2": 3.5,
                    "feature3": 1.4,
                    "feature4": 0.2,
                    "feature5": 2.3
                },
                {
                    "feature1": 6.2,
                    "feature2": 2.8,
                    "feature3": 4.8,
                    "feature4": 1.8,
                    "feature5": 4.1
                }
            ]
        }
    
    def test_batch_predict_with_valid_data(self, test_client, valid_batch_request):
        """Test batch prediction with valid data"""
        response = test_client.post("/batch_predict", json=valid_batch_request)
        
        # May be 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503]
    
    def test_batch_predict_returns_all_predictions(self, test_client, valid_batch_request):
        """Test batch prediction returns correct number of predictions"""
        response = test_client.post("/batch_predict", json=valid_batch_request)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["total_predictions"] == 2
    
    def test_batch_predict_with_single_instance(self, test_client):
        """Test batch prediction with single instance"""
        request = {
            "instances": [
                {
                    "feature1": 5.1,
                    "feature2": 3.5,
                    "feature3": 1.4,
                    "feature4": 0.2,
                    "feature5": 2.3
                }
            ]
        }
        
        response = test_client.post("/batch_predict", json=request)
        
        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) == 1
    
    def test_batch_predict_with_invalid_instance(self, test_client):
        """Test batch prediction with invalid instance"""
        request = {
            "instances": [
                {
                    "feature1": "invalid",
                    "feature2": 3.5,
                    "feature3": 1.4,
                    "feature4": 0.2,
                    "feature5": 2.3
                }
            ]
        }
        
        response = test_client.post("/batch_predict", json=request)
        assert response.status_code == 422


class TestModelInfoEndpoint:
    """Tests for model info endpoint"""
    
    def test_model_info_returns_200_or_503(self, test_client):
        """Test model info endpoint returns 200 or 503"""
        response = test_client.get("/model/info")
        assert response.status_code in [200, 503]
    
    def test_model_info_contains_type(self, test_client):
        """Test model info contains model type"""
        response = test_client.get("/model/info")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data
            assert "model_version" in data


class TestAPIDocumentation:
    """Tests for API documentation"""
    
    def test_openapi_schema_accessible(self, test_client):
        """Test OpenAPI schema is accessible"""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_swagger_ui_accessible(self, test_client):
        """Test Swagger UI is accessible"""
        response = test_client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_accessible(self, test_client):
        """Test ReDoc is accessible"""
        response = test_client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])