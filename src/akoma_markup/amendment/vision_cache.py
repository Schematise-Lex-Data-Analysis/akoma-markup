"""Caching and checkpoint system for vision-based amendment extraction.

Provides checkpointing to resume interrupted extractions and caching
to avoid re-processing pages.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
import hashlib

from .vision_schema import VisionExtractedAmendment


logger = logging.getLogger(__name__)


class VisionExtractionCache:
    """Cache for vision-based amendment extraction results."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the cache.
        
        Args:
            cache_dir: Optional custom cache directory.
                      Defaults to `.akoma_cache/vision_amendments/`.
        """
        if cache_dir is None:
            self.cache_dir = Path(".akoma_cache") / "vision_amendments"
        else:
            self.cache_dir = cache_dir
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Cache directory: {self.cache_dir}")
    
    def save_checkpoint(
        self, 
        pdf_path: Path, 
        page_num: int, 
        amendments: List[VisionExtractedAmendment]
    ) -> Path:
        """Save amendments for a specific page as a checkpoint.
        
        Args:
            pdf_path: Path to the PDF file.
            page_num: Page number (1-indexed).
            amendments: List of amendments extracted from the page.
            
        Returns:
            Path to the saved checkpoint file.
        """
        cache_key = self._get_cache_key(pdf_path)
        checkpoint_file = self.cache_dir / f"{cache_key}_page_{page_num}.json"
        
        # Prepare data for serialization
        checkpoint_data = {
            "pdf_path": str(pdf_path),
            "page_num": page_num,
            "extracted_at": datetime.now().isoformat(),
            "amendments": [am.to_dict() for am in amendments]
        }
        
        # Save to file
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved checkpoint for {pdf_path.name} page {page_num}")
        return checkpoint_file
    
    def load_checkpoint(
        self, 
        pdf_path: Path
    ) -> Dict[int, List[VisionExtractedAmendment]]:
        """Load all checkpoints for a PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Dictionary mapping page numbers to lists of amendments.
        """
        cache_key = self._get_cache_key(pdf_path)
        checkpoint_pattern = f"{cache_key}_page_*.json"
        
        amendments_by_page = {}
        
        for checkpoint_file in self.cache_dir.glob(checkpoint_pattern):
            try:
                page_data = self._load_single_checkpoint(checkpoint_file)
                if page_data:
                    page_num, amendments = page_data
                    amendments_by_page[page_num] = amendments
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
        
        logger.debug(f"Loaded {len(amendments_by_page)} checkpoints for {pdf_path.name}")
        return amendments_by_page
    
    def get_processed_pages(self, pdf_path: Path) -> Set[int]:
        """Get set of pages that have been processed and cached.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Set of page numbers that have checkpoints.
        """
        cache_key = self._get_cache_key(pdf_path)
        checkpoint_pattern = f"{cache_key}_page_*.json"
        
        processed_pages = set()
        
        for checkpoint_file in self.cache_dir.glob(checkpoint_pattern):
            try:
                # Extract page number from filename
                filename = checkpoint_file.stem
                page_part = filename.split("_page_")[-1]
                page_num = int(page_part)
                processed_pages.add(page_num)
            except (ValueError, IndexError):
                logger.warning(f"Invalid checkpoint filename: {checkpoint_file}")
        
        return processed_pages
    
    def clear_cache(self, pdf_path: Optional[Path] = None) -> int:
        """Clear cache entries.
        
        Args:
            pdf_path: If provided, clear only cache for this PDF.
                     If None, clear entire cache.
                     
        Returns:
            Number of files deleted.
        """
        if pdf_path is None:
            # Clear entire cache directory
            files_to_delete = list(self.cache_dir.glob("*.json"))
        else:
            # Clear only cache for specific PDF
            cache_key = self._get_cache_key(pdf_path)
            files_to_delete = list(self.cache_dir.glob(f"{cache_key}_page_*.json"))
        
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleared {deleted_count} cache files")
        return deleted_count
    
    def save_extraction_result(
        self,
        pdf_path: Path,
        result: Any,  # AmendmentExtractionResult or similar
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save complete extraction result.
        
        Args:
            pdf_path: Path to the PDF file.
            result: Extraction result to save.
            metadata: Additional metadata to include.
            
        Returns:
            Path to the saved result file.
        """
        cache_key = self._get_cache_key(pdf_path)
        result_file = self.cache_dir / f"{cache_key}_result.json"
        
        # Convert result to dictionary
        if hasattr(result, "to_dict"):
            result_data = result.to_dict()
        elif hasattr(result, "__dict__"):
            result_data = result.__dict__
        else:
            result_data = {"result": str(result)}
        
        # Add metadata
        if metadata:
            result_data["metadata"] = metadata
        
        result_data["saved_at"] = datetime.now().isoformat()
        
        # Save to file
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extraction result for {pdf_path.name}")
        return result_file
    
    def load_extraction_result(
        self, 
        pdf_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Load complete extraction result.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            Dictionary with extraction result, or None if not found.
        """
        cache_key = self._get_cache_key(pdf_path)
        result_file = self.cache_dir / f"{cache_key}_result.json"
        
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            logger.debug(f"Loaded extraction result for {pdf_path.name}")
            return result_data
        except Exception as e:
            logger.warning(f"Failed to load extraction result {result_file}: {e}")
            return None
    
    def get_cache_info(self, pdf_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get information about cache contents.
        
        Args:
            pdf_path: If provided, get info only for this PDF.
                     If None, get info for all cached PDFs.
                     
        Returns:
            Dictionary with cache statistics.
        """
        if pdf_path is None:
            pattern = "*_page_*.json"
        else:
            cache_key = self._get_cache_key(pdf_path)
            pattern = f"{cache_key}_page_*.json"
        
        checkpoint_files = list(self.cache_dir.glob(pattern))
        
        # Group by PDF
        pdf_stats = {}
        for file_path in checkpoint_files:
            try:
                # Extract PDF identifier from filename
                filename = file_path.stem
                if "_page_" in filename:
                    pdf_id = filename.split("_page_")[0]
                    
                    if pdf_id not in pdf_stats:
                        pdf_stats[pdf_id] = {
                            "checkpoint_count": 0,
                            "pages": set()
                        }
                    
                    pdf_stats[pdf_id]["checkpoint_count"] += 1
                    
                    # Extract page number
                    page_part = filename.split("_page_")[-1]
                    page_num = int(page_part)
                    pdf_stats[pdf_id]["pages"].add(page_num)
            except (ValueError, IndexError):
                continue
        
        # Convert sets to sorted lists
        for stats in pdf_stats.values():
            stats["pages"] = sorted(stats["pages"])
        
        total_checkpoints = len(checkpoint_files)
        total_pdfs = len(pdf_stats)
        
        return {
            "cache_dir": str(self.cache_dir),
            "total_checkpoints": total_checkpoints,
            "total_pdfs": total_pdfs,
            "pdf_stats": pdf_stats,
            "cache_size_mb": self._get_cache_size_mb()
        }
    
    def _get_cache_key(self, pdf_path: Path) -> str:
        """Generate cache key for a PDF file.
        
        Uses file path and modification time to detect changes.
        """
        # Get file stats
        stat = pdf_path.stat()
        
        # Create hash of path and modification time
        key_data = f"{pdf_path.resolve()}_{stat.st_mtime}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _load_single_checkpoint(self, checkpoint_file: Path) -> Optional[tuple[int, List[VisionExtractedAmendment]]]:
        """Load a single checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file.
            
        Returns:
            Tuple of (page_num, amendments) or None if failed.
        """
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            page_num = data["page_num"]
            
            # Deserialize amendments
            amendments = [
                VisionExtractedAmendment.from_dict(am_dict)
                for am_dict in data["amendments"]
            ]
            
            return page_num, amendments
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
            return None
    
    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in megabytes."""
        total_bytes = 0
        for file_path in self.cache_dir.glob("*.json"):
            try:
                total_bytes += file_path.stat().st_size
            except Exception:
                continue
        
        return total_bytes / (1024 * 1024)


# Convenience functions
def get_default_cache() -> VisionExtractionCache:
    """Get default cache instance."""
    return VisionExtractionCache()


def clear_pdf_cache(pdf_path: Path) -> int:
    """Clear cache for a specific PDF.
    
    Args:
        pdf_path: Path to PDF file.
        
    Returns:
        Number of files deleted.
    """
    cache = get_default_cache()
    return cache.clear_cache(pdf_path)


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache.
    
    Returns:
        Dictionary with cache statistics.
    """
    cache = get_default_cache()
    return cache.get_cache_info()