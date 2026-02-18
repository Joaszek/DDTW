"""Configuration module for DTW cost analysis."""

import yaml
from pathlib import Path
from typing import Dict, List


class Config:
    """Configuration class for loading and accessing config values."""

    def __init__(self, config_path: str = "config.yaml"):
        """Load configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Extract configuration values
        self.data_root = Path(self._config["data_root"])
        self.categories: List[str] = self._config["categories"]
        self.parallel_files: Dict[str, str] = self._config["parallel_files"]
        self.train_ratio_per_class: float = self._config["train_ratio_per_class"]
        self.val_ratio_per_class: float = self._config["val_ratio_per_class"]
        self.test_ratio_per_class: float = self._config["test_ratio_per_class"]
        self.random_state: int = self._config["random_state"]
        self.window_len: int = self._config["window_len"]
        self.window_step: int = self._config["window_step"]
        self.use_log1p: bool = self._config["use_log1p"]
        self.use_norm_by_median: bool = self._config["use_norm_by_median"]

    def __repr__(self) -> str:
        """Return string representation of config."""
        return f"Config(data_root={self.data_root}, categories={self.categories})"


# Global config instance
config = Config()
