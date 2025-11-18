"""Base model class for all models in the suite."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses."""
        pass

    def save_model(self, save_path: Union[str, Path], save_config: bool = True):
        """
        Save model weights and configuration.

        Args:
            save_path: Path to save the model
            save_config: Whether to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

        # Save configuration
        if save_config:
            config_path = save_path.parent / f"{save_path.stem}_config.pt"
            torch.save(self.config, config_path)
            logger.info(f"Config saved to {config_path}")

    def load_model(self, load_path: Union[str, Path], strict: bool = True):
        """
        Load model weights.

        Args:
            load_path: Path to load the model from
            strict: Whether to strictly enforce state dict keys match
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        state_dict = torch.load(load_path, map_location=self.device)
        self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model loaded from {load_path}")

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size(self) -> float:
        """
        Get model size in MB.

        Returns:
            Model size in megabytes
        """
        param_size = 0
        buffer_size = 0

        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def freeze_parameters(self):
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False
        logger.info("All parameters frozen")

    def unfreeze_parameters(self):
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen")

    def freeze_layer(self, layer_name: str):
        """
        Freeze specific layer.

        Args:
            layer_name: Name of layer to freeze
        """
        for name, param in self.named_parameters():
            if layer_name in name:
                param.requires_grad = False
        logger.info(f"Layer '{layer_name}' frozen")

    def unfreeze_layer(self, layer_name: str):
        """
        Unfreeze specific layer.

        Args:
            layer_name: Name of layer to unfreeze
        """
        for name, param in self.named_parameters():
            if layer_name in name:
                param.requires_grad = True
        logger.info(f"Layer '{layer_name}' unfrozen")

    def to_device(self, device: Optional[Union[str, torch.device]] = None):
        """
        Move model to device.

        Args:
            device: Target device
        """
        if device is not None:
            self.device = torch.device(device)
        self.to(self.device)
        logger.info(f"Model moved to {self.device}")

    def get_optimizer(self, optimizer_config: Dict[str, Any]) -> torch.optim.Optimizer:
        """
        Get optimizer for model.

        Args:
            optimizer_config: Optimizer configuration

        Returns:
            Configured optimizer
        """
        optimizer_type = optimizer_config.get("type", "adamw").lower()
        lr = optimizer_config.get("learning_rate", 1e-5)
        weight_decay = optimizer_config.get("weight_decay", 0.01)

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                betas=optimizer_config.get("betas", (0.9, 0.999)),
                eps=optimizer_config.get("eps", 1e-8),
                weight_decay=weight_decay,
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                betas=optimizer_config.get("betas", (0.9, 0.999)),
                eps=optimizer_config.get("eps", 1e-8),
                weight_decay=weight_decay,
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=optimizer_config.get("momentum", 0.9),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Get learning rate scheduler.

        Args:
            optimizer: Optimizer to schedule
            scheduler_config: Scheduler configuration

        Returns:
            Configured scheduler
        """
        scheduler_type = scheduler_config.get("type", "linear").lower()

        if scheduler_type == "linear":
            from torch.optim.lr_scheduler import LambdaLR

            num_warmup_steps = scheduler_config.get("num_warmup_steps", 0)
            num_training_steps = scheduler_config.get("num_training_steps", 1000)

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )

            scheduler = LambdaLR(optimizer, lr_lambda)

        elif scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get("T_max", 1000),
                eta_min=scheduler_config.get("min_lr", 0),
            )

        elif scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR

            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get("step_size", 100),
                gamma=scheduler_config.get("gamma", 0.1),
            )

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler

    def print_model_summary(self):
        """Print model summary."""
        logger.info("=" * 50)
        logger.info("Model Summary")
        logger.info("=" * 50)
        logger.info(f"Total parameters: {self.count_parameters():,}")
        logger.info(f"Model size: {self.get_model_size():.2f} MB")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 50)
