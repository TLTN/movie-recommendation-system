import yaml
from pathlib import Path
from typing import Dict, Any
import logging


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._create_directories()

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def _create_directories(self) -> None:
        paths = self.config.get('paths', {})
        directories = [
            self.config['data']['processed_path'],
            paths.get('models', 'saved_models'),
            paths.get('results', 'results'),
            paths.get('plots', 'results/plots'),
            paths.get('logs', 'logs')
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


config = Config()