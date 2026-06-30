import os

class Config:
    """Application configuration"""
    
    EXCEL_PATH = os.environ.get("EXCEL_PATH", "Indian_Foods_GI_GL_Database.xlsx")
    DEFAULT_SENSITIVITY = float(os.environ.get("DEFAULT_SENSITIVITY", 1.0))
    DEFAULT_CARBS_LIMIT = int(os.environ.get("DEFAULT_CARBS_LIMIT", 30))
    
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Clinical safety bounds
    MIN_SENSITIVITY = 0.1
    MAX_SENSITIVITY = 3.0
