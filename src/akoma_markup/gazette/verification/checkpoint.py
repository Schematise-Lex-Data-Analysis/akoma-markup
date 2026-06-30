"""Checkpoint manager for Gazette verification."""

import hashlib
import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class VerificationCheckpointManager:
    """Manages checkpoints for verification process."""
    
    CHECKPOINT_DIR = ".akoma_cache/verification"
    CHECKPOINT_TTL_DAYS = 7  # Checkpoints older than 7 days are stale
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Custom checkpoint directory (default: .akoma_cache/verification)
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path(self.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _get_checkpoint_id(self, input_path: Path, output_path: Path) -> str:
        """Generate unique checkpoint ID for input/output pair."""
        # Use file paths and modification times to create unique ID
        input_info = f"{input_path}:{input_path.stat().st_mtime if input_path.exists() else 0}"
        output_info = f"{output_path}:{output_path.stat().st_mtime if output_path.exists() else 0}"
        
        combined = f"{input_info}|{output_info}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get full path for checkpoint file."""
        return self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pkl"
    
    def save_checkpoint(
        self,
        input_path: Path,
        output_path: Path,
        checkpoint_data: Dict[str, Any],
    ) -> Path:
        """Save checkpoint data.
        
        Args:
            input_path: Path to input TSV file
            output_path: Path to output TSV file
            checkpoint_data: Data to save in checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint_id = self._get_checkpoint_id(input_path, output_path)
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        # Add metadata to checkpoint
        checkpoint_data["metadata"] = {
            "checkpoint_id": checkpoint_id,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "created_at": datetime.now().isoformat(),
            "processed_count": checkpoint_data.get("processed_count", 0),
        }
        
        # Serialize results_df separately if present
        if "results_df" in checkpoint_data and isinstance(checkpoint_data["results_df"], pd.DataFrame):
            # Convert DataFrame to dict for serialization
            df_dict = checkpoint_data["results_df"].to_dict(orient="split")
            checkpoint_data["results_df"] = df_dict
        
        try:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(checkpoint_data, f)
            
            logger.debug(f"Saved checkpoint: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            raise
    
    def load_checkpoint(
        self,
        input_path: Path,
        output_path: Path,
    ) -> Optional[Dict[str, Any]]:
        """Load checkpoint data if available and valid.
        
        Args:
            input_path: Path to input TSV file
            output_path: Path to output TSV file
            
        Returns:
            Checkpoint data if available and valid, None otherwise
        """
        checkpoint_id = self._get_checkpoint_id(input_path, output_path)
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if not checkpoint_path.exists():
            logger.debug(f"No checkpoint found: {checkpoint_path}")
            return None
        
        # Check if checkpoint is stale (older than TTL)
        try:
            stat = checkpoint_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime)
            if datetime.now() - created_at > timedelta(days=self.CHECKPOINT_TTL_DAYS):
                logger.info(f"Checkpoint is stale (created {created_at}), ignoring")
                self.clear_checkpoint(input_path, output_path)
                return None
        except Exception as e:
            logger.warning(f"Failed to check checkpoint age: {e}")
        
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
            
            # Restore DataFrame if present
            if "results_df" in checkpoint_data and isinstance(checkpoint_data["results_df"], dict):
                try:
                    df_dict = checkpoint_data["results_df"]
                    checkpoint_data["results_df"] = pd.DataFrame(
                        data=df_dict["data"],
                        columns=df_dict["columns"],
                        index=df_dict["index"] if "index" in df_dict else None,
                    )
                except Exception as e:
                    logger.error(f"Failed to restore DataFrame from checkpoint: {e}")
                    return None
            
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            # Remove corrupt checkpoint
            try:
                checkpoint_path.unlink()
                logger.info(f"Removed corrupt checkpoint: {checkpoint_path}")
            except Exception:
                pass
            return None
    
    def clear_checkpoint(self, input_path: Path, output_path: Path) -> bool:
        """Clear checkpoint for given input/output pair.
        
        Args:
            input_path: Path to input TSV file
            output_path: Path to output TSV file
            
        Returns:
            True if checkpoint was cleared, False otherwise
        """
        checkpoint_id = self._get_checkpoint_id(input_path, output_path)
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                logger.debug(f"Cleared checkpoint: {checkpoint_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear checkpoint {checkpoint_path}: {e}")
                return False
        
        return False
    
    def clear_all_checkpoints(self) -> int:
        """Clear all checkpoints in the checkpoint directory.
        
        Returns:
            Number of checkpoints cleared
        """
        cleared_count = 0
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                checkpoint_file.unlink()
                cleared_count += 1
                logger.debug(f"Cleared checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to clear checkpoint {checkpoint_file}: {e}")
        
        logger.info(f"Cleared {cleared_count} checkpoints")
        return cleared_count
    
    def cleanup_stale_checkpoints(self) -> int:
        """Remove checkpoints older than TTL.
        
        Returns:
            Number of stale checkpoints removed
        """
        removed_count = 0
        cutoff_time = datetime.now() - timedelta(days=self.CHECKPOINT_TTL_DAYS)
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                stat = checkpoint_file.stat()
                created_at = datetime.fromtimestamp(stat.st_mtime)
                
                if created_at < cutoff_time:
                    checkpoint_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed stale checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to check/remove checkpoint {checkpoint_file}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} stale checkpoints")
        
        return removed_count
    
    def get_checkpoint_info(self, input_path: Path, output_path: Path) -> Optional[Dict[str, Any]]:
        """Get information about checkpoint without loading full data.
        
        Args:
            input_path: Path to input TSV file
            output_path: Path to output TSV file
            
        Returns:
            Checkpoint metadata if exists, None otherwise
        """
        checkpoint_id = self._get_checkpoint_id(input_path, output_path)
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            stat = checkpoint_path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime)
            size_kb = stat.st_size / 1024
            
            return {
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path),
                "created_at": created_at.isoformat(),
                "age_days": (datetime.now() - created_at).days,
                "size_kb": round(size_kb, 2),
                "is_stale": (datetime.now() - created_at).days > self.CHECKPOINT_TTL_DAYS,
            }
        except Exception as e:
            logger.error(f"Failed to get checkpoint info {checkpoint_path}: {e}")
            return None