from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import os
from datetime import datetime

from engine import ClinicalOrchestrator, Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CGMRequest(BaseModel):
    cgm: List[float] = Field(..., min_items=10, max_items=10)
    carbs: Optional[int] = Field(30, ge=0, le=200)
    meal_type: Optional[str] = Field("regular")
    sensitivity: Optional[float] = Field(1.0, ge=0.5, le=2.0)
    
    @validator('cgm')
    def validate_cgm(cls, v):
        if len(v) != 10:
            raise ValueError(f"Must provide exactly 10 CGM values, got {len(v)}")
        if any(x < 20 or x > 600 for x in v):
            raise ValueError("CGM values must be between 20-600 mg/dL")
        return v

app = FastAPI(
    title="CDSS - Clinical Decision Support System",
    description="AI-powered glucose prediction with phase-aware food recommendations",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    config = Config()
    orchestrator = ClinicalOrchestrator(config)
    logger.info("✅ CDSS API initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize: {e}")
    raise

@app.get("/")
async def root():
    return {
        "status": "✅ CDSS API Running",
        "food_source": "STRICT Excel-only (Indian Foods GI/GL Database)",
        "food_count": len(orchestrator.food_recommender.df),
        "version": "4.0.0",
        "features": [
            "Glucose prediction (30/60/120 min)",
            "Risk assessment with uncertainty bounds",
            "STRICT Excel-based food recommendations",
            "Personalization sensitivity factor"
        ],
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": "cpu",
        "foods_loaded": len(orchestrator.food_recommender.df),
        "sensitivity": orchestrator.sensitivity,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: CGMRequest):
    try:
        result = orchestrator.run(
            cgm_readings=request.cgm,
            carbs=request.carbs,
            meal_type=request.meal_type,
            patient_sensitivity=request.sensitivity
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/foods")
async def get_all_foods(limit: int = 20):
    try:
        foods = orchestrator.food_recommender.get_all_foods(limit)
        return {
            "status": "success",
            "count": len(foods),
            "total_available": len(orchestrator.food_recommender.df),
            "foods": foods
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/foods/gi/{gi_band}")
async def get_foods_by_gi(gi_band: str):
    try:
        foods = orchestrator.food_recommender.get_food_by_gi_band(gi_band)
        return {
            "status": "success",
            "gi_band": gi_band,
            "count": len(foods),
            "foods": foods
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/foods/search")
async def search_foods(query: str):
    try:
        foods = orchestrator.food_recommender.search_food(query)
        return {
            "status": "success",
            "query": query,
            "count": len(foods),
            "foods": foods
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
