import json
import os
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)

DEFAULTS = {
    "live_interval": "15m",
    "research_interval": "1d",
    "div_cagr_years": 10,
    "benchmark_refresh_days": 7,
    "lsa_variance_threshold": 0.60,
    "news_max_age_days": 90,
    "output_dir": "output/",
}

STALE_THRESHOLD = {
        "1m": timedelta(minutes=1), 
        "2m": timedelta(minutes=2),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(minutes=60),
        "90m": timedelta(minutes=90),
        "1d": timedelta(days=1),
    }

INTERVAL_MAX_LOOKBACK = {
    "1m": timedelta(days=59), # safe margin under yfinance's ~60d cap for sub-daily
    "2m": timedelta(days=59),
    "5m": timedelta(days=59),
    "15m": timedelta(days=59),
    "30m": timedelta(days=59),
    "1h":  timedelta(days=59), 
    "90m": timedelta(days=59),
    "1d":  timedelta(days=365),
}

class AppConfig:
    def __init__(self, config_path: str):
        self._path = config_path
        self._data = {**DEFAULTS}
        dir_name = os.path.dirname(config_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            logger.warning(f"No config file found at {self._path}, creating with defaults.")
            self.save()  # write defaults to disk immediately
            return
        try:
            with open(self._path) as f:
                self._data.update(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Config load failed, using defaults: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, 'w') as f:
            json.dump(self._data, f, indent=4)

    def get(self, key):
        return self._data[key]

    def set(self, key: str, value):
        if key not in DEFAULTS:
            raise KeyError(f"Unknown config key '{key}'")
        self._data[key] = value
        self.save()