"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the Multimodal AI Suite."""

    def __init__(self):
        self._configs: Dict[str, Any] = {}
        self.config_dir = Path(__file__).parent.parent.parent / "configs"

    def load_config(self, config_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            config_name: Name of the config (e.g., 'model_config', 'training_config')
            config_path: Optional custom path to config file

        Returns:
            Dictionary containing configuration
        """
        if config_name in self._configs:
            return self._configs[config_name]

        if config_path is None:
            config_path = self.config_dir / f"{config_name}.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Environment variable substitution
        config = self._substitute_env_vars(config)

        self._configs[config_name] = config
        return config

    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in config values.

        Supports ${VAR_NAME} or ${VAR_NAME:default_value} syntax.
        """
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Handle ${VAR:default} syntax
            if config.startswith("${") and config.endswith("}"):
                var_expr = config[2:-1]
                if ":" in var_expr:
                    var_name, default = var_expr.split(":", 1)
                    return os.getenv(var_name, default)
                else:
                    return os.getenv(var_expr, config)
        return config

    def get(self, config_name: str, key_path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            config_name: Name of the config file
            key_path: Dot-separated path to nested key (e.g., 'model.hidden_size')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if config_name not in self._configs:
            self.load_config(config_name)

        config = self._configs[config_name]

        if key_path is None:
            return config

        # Navigate nested keys
        keys = key_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def update(self, config_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration values.

        Args:
            config_name: Name of the config to update
            updates: Dictionary of updates to apply
        """
        if config_name not in self._configs:
            self.load_config(config_name)

        self._configs[config_name].update(updates)

    def reload(self, config_name: Optional[str] = None) -> None:
        """
        Reload configuration from disk.

        Args:
            config_name: Specific config to reload, or None to reload all
        """
        if config_name:
            if config_name in self._configs:
                del self._configs[config_name]
                self.load_config(config_name)
        else:
            config_names = list(self._configs.keys())
            self._configs.clear()
            for name in config_names:
                self.load_config(name)


# Global configuration instance
_config = Config()


def load_config(config_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a configuration file."""
    return _config.load_config(config_name, config_path)


def get_config(config_name: str, key_path: Optional[str] = None, default: Any = None) -> Any:
    """Get a configuration value."""
    return _config.get(config_name, key_path, default)


def update_config(config_name: str, updates: Dict[str, Any]) -> None:
    """Update configuration values."""
    _config.update(config_name, updates)


def reload_config(config_name: Optional[str] = None) -> None:
    """Reload configuration from disk."""
    _config.reload(config_name)
