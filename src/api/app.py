"""
FastAPI Deployment Endpoint
Serves ML model predictions via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import os
from datetime import datetime


# Define request schema
class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    feature1: float = Field(..., description="Feature 1 value")
    feature2: float = Field(..., description="Feature 2 value")
    feature3: float = Field(..., description="Feature 3 value")
    feature4: float = Field(..., description="Feature 4 value")
    feature5: float = Field(..., description="Feature 5 value")
    
    class Config:
        json_schema_extra = {
            "example": {
                "feature1": 5.1,
                "feature2": 3.5,
                "feature3": 1.4,
                "feature4": 0.2,
                "feature5": 2.3
            }
        }
    
    @validator('*')
    def check_not_nan(cls, v):
        """Validate that values are not NaN"""
        if pd.isna(v):
            raise ValueError('Value cannot be NaN')
        return v


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    instances: List[PredictionRequest]
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: int
    prediction_label: str
    confidence: float
    feature_importance: Optional[dict] = None
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[PredictionResponse]
    total_predictions: int
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_loaded: bool
    model_path: str
    model_version: str
    api_version: str
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="ML Model Prediction API",
    description="REST API for serving machine learning model predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
model = None
scaler = None
model_version = "1.0.0"
model_path = "models/model.pkl"
scaler_path = "data/processed/scaler.pkl"

# Class labels (customize based on your model)
CLASS_LABELS = {
    0: "Class_0",
    1: "Class_1",
    2: "Class_2"
}


def load_model():
    """Load model and scaler at startup"""
    global model, scaler
    
    try:
        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
        else:
            print(f"⚠ Model file not found: {model_path}")
            model = None
        
        # Load scaler
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded from: {scaler_path}")
        else:
            print(f"⚠ Scaler file not found: {scaler_path}")
            scaler = None
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None
        scaler = None


def preprocess_features(data: dict) -> np.ndarray:
    """
    Preprocess features for prediction
    
    Args:
        data: Dictionary with feature values
    
    Returns:
        Preprocessed feature array
    """
    # Extract features
    features = np.array([[
        data['feature1'],
        data['feature2'],
        data['feature3'],
        data['feature4'],
        data['feature5']
    ]])
    
    # Create engineered features (same as training)
    feature_ratio = features[0][0] / (features[0][1] + 1e-10)
    feature_sum = features[0][2] + features[0][3]
    
    # Combine all features
    features_extended = np.array([[
        features[0][0],
        features[0][1],
        features[0][2],
        features[0][3],
        features[0][4],
        feature_ratio,
        feature_sum
    ]])
    
    # Scale features if scaler available
    if scaler is not None:
        features_scaled = scaler.transform(features_extended)
    else:
        features_scaled = features_extended
    
    return features_scaled


def preprocess_batch_features(instances: List[PredictionRequest]) -> np.ndarray:
    """
    Preprocess a batch of features for prediction

    Args:
        instances: List of PredictionRequest objects

    Returns:
        Preprocessed feature array (batch)
    """
    # Convert to DataFrame
    df = pd.DataFrame([i.dict() for i in instances])

    # Create engineered features (vectorized)
    df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-10)
    df['feature_sum'] = df['feature3'] + df['feature4']

    # Select columns in correct order
    features_extended = df[[
        'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
        'feature_ratio', 'feature_sum'
    ]].values

    # Scale features if scaler available
    if scaler is not None:
        features_scaled = scaler.transform(features_extended)
    else:
        features_scaled = features_extended

    return features_scaled


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "="*50)
    print("Starting ML Prediction API")
    print("="*50 + "\n")
    load_model()
    print("\n" + "="*50)
    print("API Ready!")
    print("="*50 + "\n")
    print("Access API documentation at: http://127.0.0.1:8000/docs")
    print()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns model status and API information
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=model_path,
        model_version=model_version,
        api_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction for single instance
    
    Args:
        request: PredictionRequest with feature values
    
    Returns:
        PredictionResponse with prediction and confidence
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Preprocess features
        features = preprocess_features(request.dict())
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 
                           'feature5', 'feature_ratio', 'feature_sum']
            importance_dict = {
                name: float(importance) 
                for name, importance in zip(feature_names, model.feature_importances_)
            }
            # Sort by importance
            feature_importance = dict(sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])  # Top 5 features
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=CLASS_LABELS.get(int(prediction), f"Class_{prediction}"),
            confidence=confidence,
            feature_importance=feature_importance,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple instances
    
    Args:
        request: BatchPredictionRequest with list of instances
    
    Returns:
        BatchPredictionResponse with all predictions
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Preprocess batch
        features = preprocess_batch_features(request.instances)

        # Make predictions (vectorized)
        predictions_raw = model.predict(features)

        # Get confidences (vectorized)
        confidences = np.ones(len(predictions_raw))
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
            # Efficiently select probability for predicted class
            predictions_indices = predictions_raw.astype(int)
            confidences = probabilities[np.arange(len(predictions_raw)), predictions_indices]

        # Construct response
        predictions = []
        timestamp = datetime.now().isoformat()
        
        for i, pred_val in enumerate(predictions_raw):
            pred_int = int(pred_val)
            predictions.append(PredictionResponse(
                prediction=pred_int,
                prediction_label=CLASS_LABELS.get(pred_int, f"Class_{pred_int}"),
                confidence=float(confidences[i]),
                feature_importance=None,  # Skip for batch to save time
                model_version=model_version,
                timestamp=timestamp
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            model_version=model_version,
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """
    Get model information
    
    Returns:
        Model metadata and statistics
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    info = {
        "model_type": type(model).__name__,
        "model_version": model_version,
        "model_path": model_path,
        "scaler_loaded": scaler is not None
    }
    
    # Add model-specific info
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        info['max_depth'] = model.max_depth
    if hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_
    if hasattr(model, 'n_classes_'):
        info['n_classes'] = model.n_classes_
    
    return info


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("Starting FastAPI Server")
    print("="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )