"""
Reproducibility utilities for LSTM iron ore forecasting model.

This module provides comprehensive seed setting across all random number generators
used in the ML pipeline including Python's random, NumPy, PyTorch, and CUDA operations.
"""

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(
    seed: int, deterministic: bool = True, cuda_deterministic: bool = True
) -> None:
    """
    Set random seeds for reproducible results across all components.

    This function ensures reproducible results by setting seeds for:
    - Python's built-in random module
    - NumPy random number generator
    - PyTorch CPU operations
    - PyTorch CUDA operations
    - Environment variables for additional determinism

    Args:
        seed: Random seed value (typically 42 for ML experiments)
        deterministic: Enable PyTorch deterministic algorithms (may reduce performance)
        cuda_deterministic: Enable CUDA deterministic operations (may reduce performance)

    Note:
        Enabling deterministic operations may impact training speed but ensures
        fully reproducible results across different runs and hardware.

    Example:
        >>> set_seed(42, deterministic=True, cuda_deterministic=True)
        >>> # Now all random operations will be reproducible
    """
    logger.info(f"Setting random seed to {seed} for reproducibility")

    # Set Python built-in random seed
    random.seed(seed)
    logger.debug("Set Python random seed")

    # Set NumPy random seed
    np.random.seed(seed)
    logger.debug("Set NumPy random seed")

    # Set PyTorch CPU random seed
    torch.manual_seed(seed)
    logger.debug("Set PyTorch CPU random seed")

    # Set PyTorch CUDA random seed (affects all CUDA devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        logger.debug("Set PyTorch CUDA random seed for all devices")

        if cuda_deterministic:
            # Enable CUDA deterministic operations
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logger.debug("Enabled CUDA deterministic workspace configuration")

    # Enable PyTorch deterministic algorithms
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Note: warn_only=True allows fallback to non-deterministic algorithms
        # with warnings instead of errors for better compatibility
        logger.debug("Enabled PyTorch deterministic algorithms with warnings")

    # Set additional environment variables for determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Set PYTHONHASHSEED to {seed}")

    logger.info("Reproducibility configuration completed successfully")


def get_rng_state() -> dict:
    """
    Get current state of all random number generators.

    Returns:
        Dictionary containing current RNG states for debugging/verification

    Example:
        >>> state = get_rng_state()
        >>> print(f"Python random state: {len(state['python_random'])}")
        >>> print(f"NumPy random seed set: {state['numpy_seed_set']}")
    """
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


def seed_worker(worker_id: int) -> None:
    """
    Seed function for PyTorch DataLoader workers to ensure reproducibility.

    This function should be passed to the DataLoader worker_init_fn parameter
    to ensure that each worker has a different but reproducible seed.

    Args:
        worker_id: Worker ID provided by PyTorch DataLoader

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, worker_init_fn=seed_worker)
    """
    # Get the base seed from the main process
    worker_seed = torch.initial_seed() % 2**32

    # Set seeds for this worker
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_reproducibility(config: dict) -> None:
    """
    Validate and log reproducibility configuration.

    Args:
        config: Configuration dictionary with reproducibility settings

    Raises:
        ValueError: If reproducibility configuration is invalid

    Example:
        >>> validate_reproducibility({'seed': 42, 'deterministic': True})
    """
    repro_config = config.get("reproducibility", {})

    if not repro_config:
        logger.warning(
            "No reproducibility configuration found - results may not be reproducible"
        )
        return

    seed = repro_config.get("seed")
    if seed is None:
        raise ValueError("Reproducibility seed must be specified")

    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got: {seed}")

    deterministic = repro_config.get("deterministic", False)
    cuda_deterministic = repro_config.get("cuda_deterministic", False)

    logger.info("Reproducibility configuration validated:")
    logger.info(f"  Seed: {seed}")
    logger.info(f"  Deterministic algorithms: {deterministic}")
    logger.info(f"  CUDA deterministic: {cuda_deterministic}")

    if deterministic or cuda_deterministic:
        logger.warning("Deterministic operations enabled - training may be slower")
