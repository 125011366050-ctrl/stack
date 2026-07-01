import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tabnet.tab_model import TabNetRegressor

from config import Config

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "GL", "GL_MA_5", "GL_STD_5", "GL_Diff", "GL_Slope", "GL_Acceleration",
    "CV_Glucose", "HR", "HR_MA_5", "Meal_Flag", "Calories", "Carbs",
    "Protein", "Fat", "Fiber", "Hour", "DayOfWeek", "IsNight"
]

META_FEATURE_ORDER = [
    "LSTM_30min", "LSTM_60min", "LSTM_120min",
    "TabNet_30min", "TabNet_60min", "TabNet_120min",
    "Last_Glucose", "Mean_Glucose", "Std_Glucose", "Slope",
    "Max_Glucose", "Min_Glucose", "Glucose_Range",
    "CV_Glucose", "Glucose_Variability"
]


class LSTMForecaster(nn.Module):
    """Matches stack_lstm_best.pth state_dict: 2-layer LSTM(18->128), FC(128->3)."""
    def __init__(self, input_size=18, hidden_size=128, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


class StackingPredictor:
    """Loads LSTM + TabNet base models and XGBoost meta-learners, runs full inference."""

    def __init__(self):
        self.lstm = self._load_lstm()
        self.tabnet = {h: self._load_tabnet(p) for h, p in Config.TABNET_PATHS.items()}
        self.xgb_meta = {h: self._load_xgb(p) for h, p in Config.XGB_META_PATHS.items()}
        with open(Config.STACK_CONFIG_PATH, "rb") as f:
            self.stack_config = pickle.load(f)
        logger.info("Stacking ensemble loaded: LSTM + TabNet(x3) + XGBoost(x3)")

    def _load_lstm(self):
        model = LSTMForecaster()
        state_dict = torch.load(Config.LSTM_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_tabnet(self, path):
        model = TabNetRegressor()
        model.load_model(path)
        return model

    def _load_xgb(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ==========================
    # FEATURE ENGINEERING
    # ==========================
    def build_window(self, entries: pd.DataFrame) -> pd.DataFrame:
        """
        entries: DataFrame with columns
        [Time, Glucose, HR, Meal_Flag, Carbs, Protein, Fat, Fiber, Calories]
        Rows are the user's real readings (oldest -> newest).
        Pads backward to Config.WINDOW_SIZE by repeating the oldest row.
        """
        df = entries.copy().reset_index(drop=True)
        df["Time"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
        df["Hour"] = df["Time"].dt.hour.fillna(12).astype(int)
        df["DayOfWeek"] = pd.Timestamp.now().dayofweek
        df["IsNight"] = df["Hour"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
        df = df.rename(columns={"Glucose": "GL"})

        n_missing = Config.WINDOW_SIZE - len(df)
        if n_missing > 0:
            pad_row = df.iloc[[0]]
            padding = pd.concat([pad_row] * n_missing, ignore_index=True)
            df = pd.concat([padding, df], ignore_index=True)

        df["GL_MA_5"] = df["GL"].rolling(5, min_periods=1).mean()
        df["GL_STD_5"] = df["GL"].rolling(5, min_periods=1).std().fillna(0)
        df["GL_Diff"] = df["GL"].diff().fillna(0)
        df["GL_Slope"] = df["GL_Diff"] / 5.0
        df["GL_Acceleration"] = df["GL_Diff"].diff().fillna(0)
        df["CV_Glucose"] = (df["GL"].rolling(5, min_periods=1).std() /
                             df["GL"].rolling(5, min_periods=1).mean()).fillna(0) * 100
        df["HR_MA_5"] = df["HR"].rolling(5, min_periods=1).mean()

        return df[FEATURE_ORDER]

    def _summary_stats(self, real_entries: pd.DataFrame) -> dict:
        gl = real_entries["Glucose"].astype(float)
        std = gl.std() if len(gl) > 1 else 0.0
        mean = gl.mean()
        return {
            "Last_Glucose": gl.iloc[-1],
            "Mean_Glucose": mean,
            "Std_Glucose": std,
            "Slope": (gl.iloc[-1] - gl.iloc[0]) / max(len(gl) - 1, 1),
            "Max_Glucose": gl.max(),
            "Min_Glucose": gl.min(),
            "Glucose_Range": gl.max() - gl.min(),
            "CV_Glucose": (std / mean * 100) if mean else 0.0,
            "Glucose_Variability": gl.diff().abs().mean() if len(gl) > 1 else 0.0,
        }

    # ==========================
    # INFERENCE
    # ==========================
    def predict(self, real_entries: pd.DataFrame) -> dict:
        window_df = self.build_window(real_entries)
        window_arr = window_df.values.astype(np.float32)  # (36, 18)

        # LSTM: sequence input
        lstm_input = torch.tensor(window_arr).unsqueeze(0)  # (1, 36, 18)
        with torch.no_grad():
            lstm_out = self.lstm(lstm_input).squeeze(0).numpy()  # (3,)

        # TabNet: flattened input
        flat_input = window_arr.flatten().reshape(1, -1)  # (1, 648)
        tabnet_out = {
            h: float(model.predict(flat_input)[0][0])
            for h, model in self.tabnet.items()
        }

        stats = self._summary_stats(real_entries)

        meta_vector = np.array([[
            lstm_out[0], lstm_out[1], lstm_out[2],
            tabnet_out["30min"], tabnet_out["60min"], tabnet_out["120min"],
            stats["Last_Glucose"], stats["Mean_Glucose"], stats["Std_Glucose"],
            stats["Slope"], stats["Max_Glucose"], stats["Min_Glucose"],
            stats["Glucose_Range"], stats["CV_Glucose"], stats["Glucose_Variability"],
        ]], dtype=np.float32)

        return {
            "30min": round(float(self.xgb_meta["30min"].predict(meta_vector)[0]), 1),
            "60min": round(float(self.xgb_meta["60min"].predict(meta_vector)[0]), 1),
            "120min": round(float(self.xgb_meta["120min"].predict(meta_vector)[0]), 1),
        }
