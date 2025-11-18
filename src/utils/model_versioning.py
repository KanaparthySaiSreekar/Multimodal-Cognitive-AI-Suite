"""Model versioning and rollback management."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .logger import get_logger

logger = get_logger(__name__)


class ModelVersion:
    """Represents a specific model version with metadata."""

    def __init__(
        self,
        version_id: str,
        model_path: Path,
        config: Dict,
        metrics: Optional[Dict] = None,
        timestamp: Optional[datetime] = None,
    ):
        self.version_id = version_id
        self.model_path = model_path
        self.config = config
        self.metrics = metrics or {}
        self.timestamp = timestamp or datetime.now()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "model_path": str(self.model_path),
            "config": self.config,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            model_path=Path(data["model_path"]),
            config=data["config"],
            metrics=data.get("metrics"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class ModelVersionManager:
    """Manage model versions with rollback capability."""

    def __init__(self, base_dir: str = "./model_versions"):
        """
        Initialize model version manager.

        Args:
            base_dir: Base directory for model versions
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_dir / "versions.json"
        self.current_version_file = self.base_dir / "current_version.txt"

        self.versions: Dict[str, ModelVersion] = {}
        self._load_metadata()

    def save_version(
        self,
        model,
        version_id: str,
        config: Dict,
        metrics: Optional[Dict] = None,
        set_as_current: bool = True,
    ) -> ModelVersion:
        """
        Save a new model version.

        Args:
            model: Model to save
            version_id: Unique version identifier
            config: Model configuration
            metrics: Performance metrics
            set_as_current: Whether to set as current version

        Returns:
            ModelVersion object
        """
        # Create version directory
        version_dir = self.base_dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

        # Save config
        config_path = version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save metrics if provided
        if metrics:
            metrics_path = version_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Create version object
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            config=config,
            metrics=metrics,
        )

        # Add to versions
        self.versions[version_id] = version

        # Save metadata
        self._save_metadata()

        # Set as current if requested
        if set_as_current:
            self.set_current_version(version_id)

        logger.info(f"Saved model version: {version_id}")

        return version

    def load_version(self, version_id: str, model_class):
        """
        Load a specific model version.

        Args:
            version_id: Version to load
            model_class: Model class to instantiate

        Returns:
            Loaded model instance
        """
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")

        version = self.versions[version_id]

        # Instantiate model
        model = model_class(**version.config)

        # Load weights
        state_dict = torch.load(version.model_path)
        model.load_state_dict(state_dict)

        logger.info(f"Loaded model version: {version_id}")

        return model

    def get_current_version(self) -> Optional[str]:
        """Get current version ID."""
        if not self.current_version_file.exists():
            return None

        with open(self.current_version_file, "r") as f:
            return f.read().strip()

    def set_current_version(self, version_id: str):
        """
        Set current version.

        Args:
            version_id: Version to set as current
        """
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")

        with open(self.current_version_file, "w") as f:
            f.write(version_id)

        logger.info(f"Set current version to: {version_id}")

    def rollback(self, steps: int = 1) -> Optional[str]:
        """
        Rollback to a previous version.

        Args:
            steps: Number of versions to rollback

        Returns:
            New current version ID, or None if rollback failed
        """
        current_version = self.get_current_version()

        if not current_version:
            logger.error("No current version set")
            return None

        # Get version history sorted by timestamp
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.timestamp,
            reverse=True,
        )

        # Find current version index
        current_idx = next(
            (i for i, v in enumerate(sorted_versions) if v.version_id == current_version),
            None,
        )

        if current_idx is None:
            logger.error(f"Current version not found in history: {current_version}")
            return None

        # Calculate target index
        target_idx = current_idx + steps

        if target_idx >= len(sorted_versions):
            logger.error(f"Cannot rollback {steps} steps - not enough history")
            return None

        # Get target version
        target_version = sorted_versions[target_idx]

        # Set as current
        self.set_current_version(target_version.version_id)

        logger.info(f"Rolled back from {current_version} to {target_version.version_id}")

        return target_version.version_id

    def compare_versions(
        self, version_id_1: str, version_id_2: str
    ) -> Dict:
        """
        Compare two model versions.

        Args:
            version_id_1: First version
            version_id_2: Second version

        Returns:
            Comparison dictionary
        """
        if version_id_1 not in self.versions or version_id_2 not in self.versions:
            raise ValueError("One or both versions not found")

        v1 = self.versions[version_id_1]
        v2 = self.versions[version_id_2]

        comparison = {
            "version_1": {
                "id": v1.version_id,
                "timestamp": v1.timestamp.isoformat(),
                "metrics": v1.metrics,
            },
            "version_2": {
                "id": v2.version_id,
                "timestamp": v2.timestamp.isoformat(),
                "metrics": v2.metrics,
            },
        }

        # Compare metrics
        if v1.metrics and v2.metrics:
            comparison["metric_differences"] = {}
            for key in set(v1.metrics.keys()) | set(v2.metrics.keys()):
                val1 = v1.metrics.get(key, 0)
                val2 = v2.metrics.get(key, 0)
                comparison["metric_differences"][key] = {
                    "v1": val1,
                    "v2": val2,
                    "diff": val2 - val1,
                    "pct_change": ((val2 - val1) / val1 * 100) if val1 != 0 else None,
                }

        return comparison

    def delete_version(self, version_id: str, force: bool = False):
        """
        Delete a model version.

        Args:
            version_id: Version to delete
            force: Force deletion even if current version
        """
        if version_id not in self.versions:
            raise ValueError(f"Version not found: {version_id}")

        # Prevent deletion of current version unless forced
        current = self.get_current_version()
        if version_id == current and not force:
            raise ValueError(
                f"Cannot delete current version {version_id} without force=True"
            )

        # Remove version directory
        version_dir = self.base_dir / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Remove from versions dict
        del self.versions[version_id]

        # Save metadata
        self._save_metadata()

        logger.info(f"Deleted model version: {version_id}")

    def cleanup_old_versions(self, keep_n: int = 10):
        """
        Clean up old versions, keeping only the N most recent.

        Args:
            keep_n: Number of versions to keep
        """
        current = self.get_current_version()

        # Sort by timestamp
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.timestamp,
            reverse=True,
        )

        # Keep most recent N
        to_keep = set([v.version_id for v in sorted_versions[:keep_n]])

        # Always keep current version
        if current:
            to_keep.add(current)

        # Delete others
        for version_id in list(self.versions.keys()):
            if version_id not in to_keep:
                try:
                    self.delete_version(version_id, force=False)
                except ValueError:
                    pass  # Skip current version

        logger.info(f"Cleaned up old versions, kept {len(to_keep)} versions")

    def get_version_history(self) -> List[Dict]:
        """Get version history sorted by timestamp."""
        return [
            v.to_dict()
            for v in sorted(
                self.versions.values(),
                key=lambda v: v.timestamp,
                reverse=True,
            )
        ]

    def _load_metadata(self):
        """Load version metadata from disk."""
        if not self.metadata_file.exists():
            return

        with open(self.metadata_file, "r") as f:
            data = json.load(f)

        for version_data in data:
            version = ModelVersion.from_dict(version_data)
            self.versions[version.version_id] = version

        logger.info(f"Loaded {len(self.versions)} model versions")

    def _save_metadata(self):
        """Save version metadata to disk."""
        data = [v.to_dict() for v in self.versions.values()]

        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)


class AutoRollbackManager:
    """Automatically rollback on deployment failures."""

    def __init__(
        self,
        version_manager: ModelVersionManager,
        health_check_fn: callable,
        rollback_threshold: float = 0.5,
    ):
        """
        Initialize auto-rollback manager.

        Args:
            version_manager: Model version manager
            health_check_fn: Function to check model health
            rollback_threshold: Threshold for triggering rollback
        """
        self.version_manager = version_manager
        self.health_check_fn = health_check_fn
        self.rollback_threshold = rollback_threshold

    def deploy_with_rollback(
        self,
        new_version_id: str,
        model,
        validation_samples: int = 100,
    ) -> bool:
        """
        Deploy new version with automatic rollback on failure.

        Args:
            new_version_id: New version to deploy
            model: New model to deploy
            validation_samples: Number of samples for validation

        Returns:
            True if deployment succeeded, False if rolled back
        """
        # Get current version for rollback
        previous_version = self.version_manager.get_current_version()

        logger.info(f"Deploying new version: {new_version_id}")
        logger.info(f"Previous version: {previous_version}")

        # Set new version as current
        self.version_manager.set_current_version(new_version_id)

        # Run health checks
        try:
            health_score = self.health_check_fn(model, validation_samples)

            if health_score < self.rollback_threshold:
                logger.error(
                    f"Health check failed: {health_score:.2f} < {self.rollback_threshold}"
                )
                raise ValueError("Health check failed")

            logger.info(f"Deployment successful. Health score: {health_score:.2f}")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")

            if previous_version:
                logger.info(f"Rolling back to version: {previous_version}")
                self.version_manager.set_current_version(previous_version)
            else:
                logger.error("No previous version available for rollback")

            return False
