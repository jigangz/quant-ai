"""
Experiment Tracking Utilities

Captures reproducibility metadata:
- Git SHA and branch
- Data hash (for cache/dedup)
- Config hash
- Environment info
"""

import hashlib
import logging
import subprocess
import sys
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def get_git_info() -> dict[str, Any]:
    """
    Get current git commit info.
    
    Returns:
        {
            "sha": "abc123...",
            "branch": "main",
            "dirty": True/False,
        }
    """
    result = {
        "sha": None,
        "branch": None,
        "dirty": False,
    }
    
    try:
        # Get commit SHA
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        result["sha"] = sha[:12]  # Short SHA
        
        # Get branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        result["branch"] = branch
        
        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        result["dirty"] = len(status) > 0
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.debug(f"Could not get git info: {e}")
    
    return result


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the training data for reproducibility.
    
    Uses: shape + column names + first/last row + sample of values
    This is fast but still catches most data changes.
    """
    hasher = hashlib.sha256()
    
    # Shape
    hasher.update(f"shape:{df.shape}".encode())
    
    # Column names
    hasher.update(f"cols:{sorted(df.columns.tolist())}".encode())
    
    # Date range (if exists)
    if "date" in df.columns:
        hasher.update(f"dates:{df['date'].min()}-{df['date'].max()}".encode())
    
    # First and last row (as string)
    if len(df) > 0:
        hasher.update(df.iloc[0].to_string().encode())
        hasher.update(df.iloc[-1].to_string().encode())
    
    # Sample of numeric values
    numeric_cols = df.select_dtypes(include=["number"]).columns[:5]
    for col in numeric_cols:
        hasher.update(f"{col}:mean={df[col].mean():.6f}".encode())
    
    return hasher.hexdigest()[:16]


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute a hash of the training config.
    
    Useful for caching and dedup.
    """
    import json
    
    # Sort keys for consistency
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def get_environment_info() -> dict[str, Any]:
    """
    Get Python version and key package versions.
    """
    result = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "packages": {},
    }
    
    # Key packages to track
    key_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "fastapi",
        "pydantic",
    ]
    
    for pkg in key_packages:
        try:
            import importlib.metadata
            version = importlib.metadata.version(pkg)
            result["packages"][pkg] = version
        except Exception:
            pass
    
    return result


def collect_experiment_metadata(
    request_dict: dict[str, Any],
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Collect all experiment metadata in one call.
    
    Args:
        request_dict: Training request parameters
        df: Training DataFrame (optional, for data_hash)
        
    Returns:
        Dict with git_sha, git_branch, git_dirty, data_hash, 
        config_hash, python_version, packages
    """
    git_info = get_git_info()
    env_info = get_environment_info()
    
    metadata = {
        "git_sha": git_info["sha"],
        "git_branch": git_info["branch"],
        "git_dirty": git_info["dirty"],
        "config_hash": compute_config_hash(request_dict),
        "python_version": env_info["python_version"],
        "packages": env_info["packages"],
    }
    
    if df is not None:
        metadata["data_hash"] = compute_data_hash(df)
    
    return metadata
