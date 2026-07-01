import os

class Config:
    """Streamlit CDSS Configuration"""

    # Data
    EXCEL_PATH = os.environ.get(
        "EXCEL_PATH",
        "Indian_Foods_GI_GL_Database.xlsx"
    )

    # Model behavior
    DEFAULT_SENSITIVITY = float(
        os.environ.get("DEFAULT_SENSITIVITY", 1.0)
    )

    DEFAULT_CARBS_LIMIT = int(
        os.environ.get("DEFAULT_CARBS_LIMIT", 30)
    )

    # Logging (optional)
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Clinical safety bounds
    MIN_SENSITIVITY = 0.1
    MAX_SENSITIVITY = 3.0
