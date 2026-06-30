from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import os
from datetime import datetime

# Import your engine
from engine import ClinicalOrchestrator, Config

# ==============================
# LOGGING
# ==============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================
# PYDANTIC MODELS (Request/Response)
# ==============================

class CGMRequest(BaseModel):
    """
    Request model for CGM prediction
    """
    cgm: List[float] = Field(
        ..., 
        min_items=10, 
        max_items=10,
        description="10 consecutive CGM readings in mg/dL"
    )
    carbs: Optional[int] = Field(
        30, 
        ge=0, 
        le=200,
        description="Estimated carbohydrate intake in grams"
    )
    meal_type: Optional[str] = Field(
        "regular",
        description="Type of meal: regular, light, heavy"
    )
    sensitivity: Optional[float] = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Patient-specific sensitivity factor"
    )
    
    @validator('cgm')
    def validate_cgm(cls, v):
        """Validate CGM readings"""
        if len(v) != 10:
            raise ValueError(f"Must provide exactly 10 CGM values, got {len(v)}")
        if any(x < 20 or x > 600 for x in v):
            raise ValueError("CGM values must be between 20-600 mg/dL")
        return v
    
    @validator('meal_type')
    def validate_meal_type(cls, v):
        """Validate meal type"""
        valid_types = ["regular", "light", "heavy", "snack", "fasting"]
        if v.lower() not in valid_types:
            raise ValueError(f"meal_type must be one of: {valid_types}")
        return v.lower()

class FoodItem(BaseModel):
    """Food item model"""
    Food_Name: str
    GI: float
    GL: float
    Carbs_g: float
    Protein_g: float
    Fat_g: float
    Calories_kcal: float
    food_class: str
    nutrition_score: float
    explanation: Optional[Dict[str, Any]] = None

class GlucoseContext(BaseModel):
    """Glucose context model"""
    current: float
    predicted_peak: float
    predicted_trough: float
    raw_peak_upper: float
    raw_trough_lower: float
    trend: str
    trend_slope: float
    phase: str
    phase_id: str
    risk_score: float
    risk_level: str
    hypo_risk: Optional[float] = None
    hyper_risk: Optional[float] = None

class FoodRecommendationResponse(BaseModel):
    """Complete response model"""
    strategy: str
    phase: str
    phase_id: str
    urgency: str
    action_level: str
    clinical_action_score: float
    base_risk_score: float
    risk_multiplier: float
    message: str
    meal_timing: str
    clinical_reason: str
    clinical_state: str
    personalization_sensitivity: float
    glucose_context: GlucoseContext
    filters_applied: Dict[str, Any]
    foods: List[FoodItem]
    total_recommendations: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    device: str
    foods_loaded: int
    models_loaded: bool
    sensitivity: float
    timestamp: str

# ==============================
# APP INIT
# ==============================
app = FastAPI(
    title="CDSS - Clinical Decision Support System",
    description="""
    ## AI-Powered Clinical Decision Support System
    
    ### Features:
    - **Glucose Prediction**: LSTM + TabNet + XGBoost stacking
    - **Phase Detection**: Rising, Falling, Peak, Stable
    - **Bidirectional CDSS**: Handles both hypo and hyper risks
    - **Explainability**: Counterfactual reasoning for food recommendations
    - **Personalization**: Patient-specific sensitivity factor
    
    ### Data Sources:
    - CGM Readings (10 values)
    - Indian Foods GI/GL Database (Excel)
    - Patient-specific parameters
    """,
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# INITIALIZE ORCHESTRATOR
# ==============================
try:
    config = Config()
    orchestrator = ClinicalOrchestrator(config)
    logger.info("✅ CDSS API initialized successfully")
    logger.info(f"📊 Foods loaded: {len(orchestrator.food_recommender.df)}")
    logger.info(f"🖥️  Device: {config.device}")
except FileNotFoundError as e:
    logger.error(f"❌ CRITICAL: {e}")
    raise RuntimeError(f"Failed to initialize: {e}")
except Exception as e:
    logger.error(f"❌ Failed to initialize: {e}")
    raise RuntimeError(f"Failed to initialize: {e}")

# ==============================
# ENDPOINTS
# ==============================

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint - API information"""
    return {
        "status": "✅ CDSS API Running",
        "model": "LSTM + TabNet + XGBoost Stacking",
        "food_source": "STRICT Excel-only (Indian Foods GI/GL Database)",
        "food_count": len(orchestrator.food_recommender.df),
        "version": "4.0.0",
        "features": [
            "Bidirectional phase detection (Rising/Falling/Peak/Stable)",
            "Glucose prediction (30/60/120 min)",
            "Risk assessment with uncertainty bounds",
            "STRICT Excel-based food recommendations",
            "Counterfactual explainability",
            "Personalization sensitivity factor"
        ],
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        device=config.device,
        foods_loaded=len(orchestrator.food_recommender.df),
        models_loaded=True,
        sensitivity=orchestrator.sensitivity,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=FoodRecommendationResponse)
async def predict(request: CGMRequest):
    """
    Main prediction endpoint
    
    ### Input:
    - **cgm**: List of 10 CGM readings (mg/dL)
    - **carbs**: Estimated carbohydrate intake (g)
    - **meal_type**: Type of meal (regular/light/heavy/snack/fasting)
    - **sensitivity**: Patient-specific sensitivity factor (0.5-2.0)
    
    ### Output:
    - **Predictions**: 30/60/120 minute glucose forecasts
    - **Risk Assessment**: Level, type, phase, urgency
    - **Food Recommendations**: Excel-based with explanations
    - **Counterfactual Reasoning**: Why foods were selected/rejected
    """
    try:
        logger.info(f"📊 Prediction request received")
        logger.info(f"   CGM: {request.cgm}")
        logger.info(f"   Carbs: {request.carbs}g")
        logger.info(f"   Meal: {request.meal_type}")
        logger.info(f"   Sensitivity: {request.sensitivity}")
        
        # Run prediction
        result = orchestrator.run(
            cgm_readings=request.cgm,
            carbs=request.carbs,
            meal_type=request.meal_type,
            patient_sensitivity=request.sensitivity
        )
        
        # Check for errors
        if "error" in result:
            logger.error(f"❌ Prediction error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Add timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = datetime.now().isoformat()
        
        logger.info(f"✅ Prediction successful")
        logger.info(f"   Phase: {result.get('phase', 'unknown')}")
        logger.info(f"   Risk Level: {result.get('risk', {}).get('level', 'unknown')}")
        logger.info(f"   Foods: {result.get('total_recommendations', 0)}")
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/foods")
async def get_all_foods(limit: int = 20):
    """
    Get all available foods from the Excel database
    """
    try:
        foods = orchestrator.food_recommender.get_all_foods(limit)
        return {
            "status": "success",
            "count": len(foods),
            "total_available": len(orchestrator.food_recommender.df),
            "foods": foods
        }
    except Exception as e:
        logger.error(f"❌ Failed to get foods: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/foods/gi/{gi_band}")
async def get_foods_by_gi(gi_band: str):
    """
    Get foods by GI band (low/medium/high)
    """
    try:
        foods = orchestrator.food_recommender.get_food_by_gi_band(gi_band)
        return {
            "status": "success",
            "gi_band": gi_band,
            "count": len(foods),
            "foods": foods
        }
    except Exception as e:
        logger.error(f"❌ Failed to get foods by GI: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/foods/search")
async def search_foods(query: str):
    """
    Search for foods by name
    """
    try:
        foods = orchestrator.food_recommender.search_food(query)
        return {
            "status": "success",
            "query": query,
            "count": len(foods),
            "foods": foods
        }
    except Exception as e:
        logger.error(f"❌ Failed to search foods: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/foods/recommend")
async def get_recommendation(request: CGMRequest):
    """
    Get food recommendations only (without full prediction)
    """
    try:
        # Run prediction
        result = orchestrator.run(
            cgm_readings=request.cgm,
            carbs=request.carbs,
            meal_type=request.meal_type,
            patient_sensitivity=request.sensitivity
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Return only food recommendations
        return {
            "status": "success",
            "phase": result.get("phase", "unknown"),
            "phase_id": result.get("phase_id", "unknown"),
            "urgency": result.get("urgency", "NORMAL"),
            "message": result.get("message", ""),
            "foods": result.get("foods", []),
            "total_recommendations": result.get("total_recommendations", 0),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/settings/sensitivity")
async def update_sensitivity(sensitivity: float):
    """
    Update patient-specific sensitivity factor
    """
    try:
        orchestrator.set_patient_sensitivity(sensitivity)
        return {
            "status": "success",
            "sensitivity": sensitivity,
            "message": f"Sensitivity updated to {sensitivity}",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to update sensitivity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        df = orchestrator.food_recommender.df
        
        return {
            "status": "success",
            "total_foods": len(df),
            "food_classes": df["food_class"].value_counts().to_dict(),
            "fast_absorbing": int(df["fast_absorbing"].sum()),
            "slow_absorbing": int(df["slow_absorbing"].sum()),
            "avg_gi": float(df["GI"].mean()),
            "avg_carbs": float(df["Carbs (g)"].mean()),
            "sensitivity": orchestrator.sensitivity,
            "device": config.device,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# ERROR HANDLERS
# ==============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "status": "error",
        "code": exc.status_code,
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "status": "error",
        "code": 500,
        "detail": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }

# ==============================
# MAIN ENTRY POINT
# ==============================

if __name__ == "__main__":
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )
