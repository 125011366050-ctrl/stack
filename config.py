import os

class Config:
    """Streamlit CDSS Configuration"""

    # Food database
    EXCEL_PATH = os.environ.get(
        "EXCEL_PATH",
         "Indian_Foods_GI_GL_Database (1).xlsx"
    )

    # Model artifacts directory (flat repo root)
    MODEL_DIR = os.environ.get("MODEL_DIR", ".")

    LSTM_PATH = os.path.join(MODEL_DIR, "stack_lstm_best.pth")
    TABNET_PATHS = {
        "30min": os.path.join(MODEL_DIR, "stack_tabnet_30min.zip"),
        "60min": os.path.join(MODEL_DIR, "stack_tabnet_60min.zip"),
        "120min": os.path.join(MODEL_DIR, "stack_tabnet_120min.zip"),
    }
    XGB_META_PATHS = {
        "30min": os.path.join(MODEL_DIR, "stack_meta_xgb_30min.pkl"),
        "60min": os.path.join(MODEL_DIR, "stack_meta_xgb_60min.pkl"),
        "120min": os.path.join(MODEL_DIR, "stack_meta_xgb_120min.pkl"),
    }
    STACK_CONFIG_PATH = os.path.join(MODEL_DIR, "stacking_model_config.pkl")

    WINDOW_SIZE = 36  # timesteps required by LSTM/TabNet
    N_FEATURES = 18

    # Model behavior
    DEFAULT_SENSITIVITY = float(os.environ.get("DEFAULT_SENSITIVITY", 1.0))
    DEFAULT_CARBS_LIMIT = int(os.environ.get("DEFAULT_CARBS_LIMIT", 30))

    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    MIN_SENSITIVITY = 0.1
    MAX_SENSITIVITY = 3.0
