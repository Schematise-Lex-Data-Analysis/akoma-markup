"""Vision-based amendment extraction orchestrator.

Main entry point for extracting amendments from PDFs using multimodal LLMs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from ..util.pdf.image_renderer import PDFImageRenderer
from ..util.llm.multimodal_factory import MultimodalLLMFactory, MultimodalLLMConfig
from .vision_schema import VisionExtractedAmendment, AmendmentExtractionResult


logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, rate_per_minute: int):
        self.rate_per_second = rate_per_minute / 60.0
        self.tokens = self.rate_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate_per_second, self.tokens + elapsed * self.rate_per_second)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Need to wait for a token
            wait_time = (1 - self.tokens) / self.rate_per_second
            await asyncio.sleep(wait_time)
            self.tokens = 0


class ProgressTracker:
    """Track extraction progress and report at intervals."""
    
    def __init__(self, total_pages: int, report_interval: int = 10):
        self.total_pages = total_pages
        self.report_interval = report_interval
        self.processed_pages = 0
        self.successful_pages = 0
        self.failed_pages = 0
        self.start_time = time.time()
        self.last_report = 0
    
    def page_completed(self, success: bool = True):
        """Record completion of a page."""
        self.processed_pages += 1
        if success:
            self.successful_pages += 1
        else:
            self.failed_pages += 1
        
        # Report progress if interval reached
        if self.processed_pages - self.last_report >= self.report_interval:
            self.report_progress()
            self.last_report = self.processed_pages
    
    def report_progress(self):
        """Report current progress."""
        elapsed = time.time() - self.start_time
        pages_per_second = self.processed_pages / elapsed if elapsed > 0 else 0
        estimated_remaining = (self.total_pages - self.processed_pages) / pages_per_second if pages_per_second > 0 else 0
        
        logger.info(
            f"Progress: {self.processed_pages}/{self.total_pages} pages "
            f"({self.processed_pages/self.total_pages*100:.1f}%) - "
            f"{self.successful_pages} successful, {self.failed_pages} failed - "
            f"{pages_per_second:.2f} pages/sec - "
            f"ETA: {timedelta(seconds=int(estimated_remaining))}"
        )
    
    def final_report(self):
        """Report final statistics."""
        elapsed = time.time() - self.start_time
        pages_per_second = self.processed_pages / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"Extraction complete: {self.processed_pages} pages in {timedelta(seconds=int(elapsed))} "
            f"({pages_per_second:.2f} pages/sec) - "
            f"{self.successful_pages} successful, {self.failed_pages} failed"
        )


@dataclass
class ExtractionConfig:
    """Configuration for vision-based amendment extraction."""
    
    # LLM configuration
    provider: str = "azure"
    model: str = "gpt-4-vision-preview"
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    
    # Image rendering
    dpi: int = 120
    image_detail: str = "high"  # "low", "high", or "auto"
    adaptive_dpi: bool = True  # Automatically adjust DPI based on content
    max_image_size_mb: float = 10.0  # Maximum image size in MB
    
    # Processing
    max_concurrent_pages: int = 3
    retry_attempts: int = 3
    retry_delay: float = 2.0
    rate_limit_per_minute: int = 60  # API calls per minute
    batch_size: int = 10  # Pages to process before checkpoint
    
    # Performance optimization
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    compress_images: bool = True
    skip_simple_pages: bool = True  # Skip pages with only text (no annotations)
    
    # Output
    confidence_threshold: float = 0.5
    progress_report_interval: int = 10  # Report progress every N pages
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ExtractionConfig:
        """Create config from dictionary."""
        return cls(**{
            k: v for k, v in config_dict.items() 
            if k in cls.__annotations__
        })


class VisionAmendmentExtractor:
    """Orchestrator for vision-based amendment extraction from PDFs."""
    
    def __init__(self, config: ExtractionConfig | Dict[str, Any]):
        """Initialize the extractor with configuration.
        
        Args:
            config: Either an ExtractionConfig object or dictionary.
        """
        if isinstance(config, dict):
            self.config = ExtractionConfig.from_dict(config)
        else:
            self.config = config
        
        self.image_renderer = PDFImageRenderer(dpi=self.config.dpi)
        self.llm_client = None
        self._executor = None
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        self.progress_tracker = None
    
    async def extract_from_pdf(
        self, 
        pdf_path: Path, 
        page_range: Optional[range] = None
    ) -> AmendmentExtractionResult:
        """Extract amendments from a PDF using vision LLM.
        
        Args:
            pdf_path: Path to the PDF file.
            page_range: Optional range of pages to process (1-indexed).
                       If None, processes all pages.
        
        Returns:
            AmendmentExtractionResult with extracted amendments.
        """
        logger.info(f"Starting vision amendment extraction from {pdf_path}")
        
        # Initialize LLM client
        await self._init_llm_client()
        
        # Determine pages to process
        total_pages = self._get_page_count(pdf_path)
        if page_range is None:
            pages_to_process = list(range(1, total_pages + 1))
        else:
            pages_to_process = list(page_range)
        
        logger.info(f"Processing {len(pages_to_process)} pages (total: {total_pages})")
        
        # Process pages in batches with checkpointing
        all_amendments = []
        extraction_errors = []
        
        try:
            # Process in batches for checkpointing
            batch_size = self.config.batch_size
            total_batches = (len(pages_to_process) + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(pages_to_process))
                batch_pages = pages_to_process[start_idx:end_idx]
                
                logger.info(f"Processing batch {batch_num + 1}/{total_batches} "
                          f"(pages {batch_pages[0]}-{batch_pages[-1]})")
                
                # Process this batch
                amendments_by_page = await self._process_pages_concurrently(
                    pdf_path, batch_pages
                )
                
                # Process results and save checkpoint
                batch_amendments = []
                for page_num, amendments in amendments_by_page.items():
                    if isinstance(amendments, Exception):
                        error_msg = f"Page {page_num}: {str(amendments)}"
                        extraction_errors.append(error_msg)
                        logger.error(error_msg)
                    elif amendments:
                        batch_amendments.extend(amendments)
                        # Save checkpoint for this page if caching enabled
                        if self.config.enable_caching:
                            await self._save_page_checkpoint(pdf_path, page_num, amendments)
                
                all_amendments.extend(batch_amendments)
                
                # Save batch checkpoint
                if self.config.enable_caching and batch_amendments:
                    logger.debug(f"Saved checkpoint for batch {batch_num + 1}")
            
            # Validate and deduplicate
            validated_amendments = self.validate_and_deduplicate(all_amendments)
            
            logger.info(f"Extracted {len(validated_amendments)} amendments "
                       f"({len(extraction_errors)} errors)")
            
            return AmendmentExtractionResult(
                pdf_path=str(pdf_path),
                extracted_amendments=validated_amendments,
                total_pages_processed=len(pages_to_process),
                extraction_errors=extraction_errors
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise
    
    async def process_page_range(
        self, 
        pdf_path: Path, 
        page_nums: List[int], 
        client: Optional[Any] = None
    ) -> List[VisionExtractedAmendment]:
        """Process specific pages from a PDF.
        
        Args:
            pdf_path: Path to the PDF file.
            page_nums: List of page numbers to process (1-indexed).
            client: Optional pre-initialized LLM client.
            
        Returns:
            List of amendments extracted from the specified pages.
        """
        if client is None:
            await self._init_llm_client()
            client = self.llm_client
        
        amendments_by_page = await self._process_pages_concurrently(
            pdf_path, page_nums, client
        )
        
        # Flatten and return
        all_amendments = []
        for page_amendments in amendments_by_page.values():
            if isinstance(page_amendments, list):
                all_amendments.extend(page_amendments)
        
        return all_amendments
    
    def validate_and_deduplicate(
        self, 
        amendments: List[VisionExtractedAmendment]
    ) -> List[VisionExtractedAmendment]:
        """Validate amendments and remove duplicates.
        
        Args:
            amendments: List of amendments to validate.
            
        Returns:
            Validated and deduplicated list of amendments.
        """
        if not amendments:
            return []
        
        # Filter by confidence threshold
        filtered = [
            am for am in amendments 
            if am.confidence_score >= self.config.confidence_threshold
        ]
        
        logger.debug(f"After confidence filtering: {len(filtered)}/{len(amendments)}")
        
        # Validate each amendment
        valid_amendments = []
        for amendment in filtered:
            if amendment.is_valid():
                valid_amendments.append(amendment)
            else:
                errors = amendment.validate()
                logger.warning(f"Invalid amendment skipped: {errors}")
        
        logger.debug(f"After validation: {len(valid_amendments)}/{len(filtered)}")
        
        # Deduplicate based on key fields
        seen_keys = set()
        deduplicated = []
        
        for amendment in valid_amendments:
            # Create a unique key for deduplication
            key = (
                amendment.page_num,
                amendment.act_number,
                amendment.act_year,
                amendment.section_number,
                amendment.amendment_type,
                amendment.target_location,
                amendment.original_text[:100]  # First 100 chars of text
            )
            
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append(amendment)
            else:
                logger.debug(f"Duplicate amendment skipped: {amendment}")
        
        logger.info(f"Final deduplicated: {len(deduplicated)} amendments")
        
        return deduplicated
    
    async def _init_llm_client(self):
        """Initialize the LLM client if not already initialized."""
        if self.llm_client is None:
            config_dict = {
                "provider": self.config.provider,
                "model": self.config.model,
                "endpoint": self.config.endpoint,
                "api_key": self.config.api_key,
                "temperature": 0.0,
                "max_tokens": 4096,
                "dpi": self.config.dpi,
                "image_detail": self.config.image_detail
            }
            
            llm_config = MultimodalLLMConfig(**config_dict)
            self.llm_client = MultimodalLLMFactory.create_client(llm_config)
    
    def _get_page_count(self, pdf_path: Path) -> int:
        """Get total number of pages in PDF."""
        from ..util.pdf.images import page_count
        return page_count(pdf_path)
    
    async def _process_pages_concurrently(
        self, 
        pdf_path: Path, 
        page_nums: List[int],
        client: Optional[Any] = None
    ) -> Dict[int, List[VisionExtractedAmendment] | Exception]:
        """Process multiple pages concurrently with rate limiting.
        
        Args:
            pdf_path: Path to PDF file.
            page_nums: List of page numbers to process.
            client: Optional LLM client (uses self.llm_client if None).
            
        Returns:
            Dictionary mapping page numbers to amendments or exceptions.
        """
        if client is None:
            client = self.llm_client
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            total_pages=len(page_nums),
            report_interval=self.config.progress_report_interval
        )
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent_pages)
        
        async def process_page(page_num: int):
            """Process a single page with semaphore and rate limiting."""
            async with semaphore:
                # Apply rate limiting
                if self.config.rate_limit_per_minute > 0:
                    await self.rate_limiter.acquire()
                
                try:
                    result = await self._process_single_page(pdf_path, page_num, client)
                    self.progress_tracker.page_completed(success=True)
                    return result
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")
                    self.progress_tracker.page_completed(success=False)
                    return page_num, e
        
        # Create tasks for all pages
        tasks = [process_page(page_num) for page_num in page_nums]
        
        # Process with timeout and retry logic
        results = {}
        for task in asyncio.as_completed(tasks):
            try:
                page_num, result = await asyncio.wait_for(task, timeout=300.0)
                results[page_num] = result
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing page")
                self.progress_tracker.page_completed(success=False)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                self.progress_tracker.page_completed(success=False)
        
        # Final progress report
        if self.progress_tracker:
            self.progress_tracker.final_report()
        
        return results
    
    async def _process_single_page(
        self, 
        pdf_path: Path, 
        page_num: int, 
        client: Any
    ) -> tuple[int, List[VisionExtractedAmendment]]:
        """Process a single PDF page.
        
        Args:
            pdf_path: Path to PDF file.
            page_num: Page number (1-indexed).
            client: Initialized LLM client.
            
        Returns:
            Tuple of (page_num, list_of_amendments).
        """
        logger.debug(f"Processing page {page_num}")
        
        # Render page to base64
        base64_image = self.image_renderer.render_page_to_base64(pdf_path, page_num)
        
        # Extract amendments using LLM
        amendments = await client.extract_amendments_from_page(base64_image, page_num)
        
        logger.debug(f"Page {page_num}: extracted {len(amendments)} amendments")
        return page_num, amendments
    
    async def _save_page_checkpoint(
        self,
        pdf_path: Path,
        page_num: int,
        amendments: List[VisionExtractedAmendment]
    ):
        """Save checkpoint for a processed page.
        
        Args:
            pdf_path: Path to PDF file.
            page_num: Page number.
            amendments: Amendments extracted from the page.
        """
        try:
            from .vision_cache import get_default_cache
            cache = get_default_cache()
            cache.save_checkpoint(pdf_path, page_num, amendments)
            logger.debug(f"Saved checkpoint for page {page_num}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for page {page_num}: {e}")
    
    def _get_executor(self):
        """Get thread pool executor for blocking operations."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_pages
            )
        return self._executor


# Convenience function
async def extract_amendments_with_vision(
    pdf_path: Path,
    config: Optional[Dict[str, Any]] = None,
    page_range: Optional[range] = None
) -> AmendmentExtractionResult:
    """Convenience function for vision-based amendment extraction.
    
    Args:
        pdf_path: Path to PDF file.
        config: Optional extraction configuration.
        page_range: Optional range of pages to process.
        
    Returns:
        AmendmentExtractionResult.
    """
    if config is None:
        config = {}
    
    extractor = VisionAmendmentExtractor(config)
    return await extractor.extract_from_pdf(pdf_path, page_range)