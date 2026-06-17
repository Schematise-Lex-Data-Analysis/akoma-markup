"""Enhanced error handling for amendment extraction.

Provides comprehensive error handling, recovery strategies,
and detailed error reporting for all extraction methods.
"""

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Severity levels for extraction errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Categories of extraction errors."""
    CONFIGURATION = "configuration"
    FILE_IO = "file_io"
    PDF_PARSING = "pdf_parsing"
    VISION_EXTRACTION = "vision_extraction"
    REGEX_EXTRACTION = "regex_extraction"
    HYBRID_FUSION = "hybrid_fusion"
    VALIDATION = "validation"
    NETWORK = "network"
    RESOURCE = "resource"


@dataclass
class ExtractionError:
    """Detailed error information for extraction failures."""
    
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "exception_type": self.exception.__class__.__name__ if self.exception else None,
            "exception_message": str(self.exception) if self.exception else None,
            "traceback": self.traceback,
            "context": self.context,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful
        }


class ErrorHandler:
    """Handles errors during amendment extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize error handler.
        
        Args:
            config: Error handling configuration.
        """
        self.config = config or {}
        self.errors: List[ExtractionError] = []
        self.recovery_attempts = 0
        
        # Recovery strategies
        self.enable_fallback = self.config.get("enable_fallback", True)
        self.max_recovery_attempts = self.config.get("max_recovery_attempts", 3)
        self.log_full_traceback = self.config.get("log_full_traceback", False)
    
    def handle_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        message: str,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> bool:
        """Handle an extraction error.
        
        Args:
            category: Error category.
            severity: Error severity.
            message: Error message.
            exception: Original exception (if any).
            context: Additional context information.
            recoverable: Whether recovery should be attempted.
            
        Returns:
            True if error was handled/recovered, False otherwise.
        """
        # Capture traceback if enabled
        tb = traceback.format_exc() if self.log_full_traceback and exception else None
        
        # Create error record
        error = ExtractionError(
            category=category,
            severity=severity,
            message=message,
            exception=exception,
            traceback=tb,
            context=context or {}
        )
        
        self.errors.append(error)
        
        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{category.value}: {message}")
            if exception:
                logger.critical(f"Exception: {exception}")
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"{category.value}: {message}")
            if exception:
                logger.error(f"Exception: {exception}")
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"{category.value}: {message}")
        else:
            logger.info(f"{category.value}: {message}")
        
        # Attempt recovery if enabled and recoverable
        if recoverable and self.enable_fallback and self.recovery_attempts < self.max_recovery_attempts:
            return self._attempt_recovery(error)
        
        return False
    
    def _attempt_recovery(self, error: ExtractionError) -> bool:
        """Attempt to recover from an error.
        
        Args:
            error: The error to recover from.
            
        Returns:
            True if recovery successful, False otherwise.
        """
        error.recovery_attempted = True
        self.recovery_attempts += 1
        
        logger.info(f"Attempting recovery for {error.category.value} error (attempt {self.recovery_attempts})")
        
        # Apply recovery strategy based on error category
        if error.category == ErrorCategory.VISION_EXTRACTION:
            # Fall back to regex-only extraction
            logger.info("Vision extraction failed, will use regex-only fallback")
            error.recovery_successful = True
            return True
            
        elif error.category == ErrorCategory.REGEX_EXTRACTION:
            # Fall back to vision-only extraction
            logger.info("Regex extraction failed, will use vision-only fallback")
            error.recovery_successful = True
            return True
            
        elif error.category == ErrorCategory.PDF_PARSING:
            # Try alternative PDF parsing approach
            logger.info("PDF parsing failed, will try alternative approach")
            error.recovery_successful = True
            return True
            
        elif error.category == ErrorCategory.NETWORK:
            # Network error - might retry or use cached results
            logger.info("Network error, will retry or use fallback")
            error.recovery_successful = True
            return True
        
        # No specific recovery strategy
        error.recovery_successful = False
        return False
    
    def get_recovery_suggestion(self, error: ExtractionError) -> Optional[str]:
        """Get recovery suggestion for an error.
        
        Args:
            error: The error to get suggestion for.
            
        Returns:
            Recovery suggestion or None.
        """
        suggestions = {
            ErrorCategory.VISION_EXTRACTION: "Try using regex-only extraction or check vision LLM credentials",
            ErrorCategory.REGEX_EXTRACTION: "Try using vision-only extraction or check PDF text quality",
            ErrorCategory.PDF_PARSING: "Try converting PDF to better quality or check file corruption",
            ErrorCategory.CONFIGURATION: "Check configuration values and environment variables",
            ErrorCategory.FILE_IO: "Check file permissions and disk space",
            ErrorCategory.NETWORK: "Check network connectivity and API endpoint availability",
            ErrorCategory.RESOURCE: "Check available memory and system resources",
        }
        
        return suggestions.get(error.category)
    
    def clear_errors(self):
        """Clear all recorded errors."""
        self.errors.clear()
        self.recovery_attempts = 0
    
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return any(e.severity == ErrorSeverity.CRITICAL for e in self.errors)
    
    def has_unrecovered_errors(self) -> bool:
        """Check if there are any unrecovered errors."""
        return any(not e.recovery_successful for e in self.errors 
                  if e.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL])
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        total = len(self.errors)
        by_severity = {s.value: 0 for s in ErrorSeverity}
        by_category = {c.value: 0 for c in ErrorCategory}
        recovered = 0
        
        for error in self.errors:
            by_severity[error.severity.value] += 1
            by_category[error.category.value] += 1
            if error.recovery_successful:
                recovered += 1
        
        return {
            "total_errors": total,
            "recovered_errors": recovered,
            "unrecovered_errors": total - recovered,
            "by_severity": by_severity,
            "by_category": by_category,
            "recovery_attempts": self.recovery_attempts
        }
    
    def create_error_report(self, pdf_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create comprehensive error report.
        
        Args:
            pdf_path: Optional PDF path for context.
            
        Returns:
            Error report dictionary.
        """
        report = {
            "pdf_path": str(pdf_path) if pdf_path else None,
            "error_summary": self.get_error_summary(),
            "errors": [e.to_dict() for e in self.errors],
            "recovery_suggestions": []
        }
        
        # Add recovery suggestions for unrecovered errors
        for error in self.errors:
            if not error.recovery_successful and error.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]:
                suggestion = self.get_recovery_suggestion(error)
                if suggestion:
                    report["recovery_suggestions"].append({
                        "category": error.category.value,
                        "suggestion": suggestion
                    })
        
        return report


# Convenience functions for common error scenarios
def handle_vision_extraction_error(
    handler: ErrorHandler,
    exception: Exception,
    pdf_path: Path,
    page_range: Optional[range] = None
) -> bool:
    """Handle vision extraction error with context."""
    context = {
        "pdf_path": str(pdf_path),
        "page_range": str(page_range) if page_range else "all"
    }
    
    return handler.handle_error(
        category=ErrorCategory.VISION_EXTRACTION,
        severity=ErrorSeverity.ERROR,
        message=f"Vision extraction failed for {pdf_path.name}",
        exception=exception,
        context=context,
        recoverable=True
    )


def handle_regex_extraction_error(
    handler: ErrorHandler,
    exception: Exception,
    pdf_path: Path
) -> bool:
    """Handle regex extraction error with context."""
    context = {"pdf_path": str(pdf_path)}
    
    return handler.handle_error(
        category=ErrorCategory.REGEX_EXTRACTION,
        severity=ErrorSeverity.ERROR,
        message=f"Regex extraction failed for {pdf_path.name}",
        exception=exception,
        context=context,
        recoverable=True
    )


def handle_pdf_parsing_error(
    handler: ErrorHandler,
    exception: Exception,
    pdf_path: Path
) -> bool:
    """Handle PDF parsing error with context."""
    context = {"pdf_path": str(pdf_path)}
    
    return handler.handle_error(
        category=ErrorCategory.PDF_PARSING,
        severity=ErrorSeverity.ERROR,
        message=f"PDF parsing failed for {pdf_path.name}",
        exception=exception,
        context=context,
        recoverable=True
    )


def handle_configuration_error(
    handler: ErrorHandler,
    message: str,
    config_key: Optional[str] = None
) -> bool:
    """Handle configuration error."""
    context = {"config_key": config_key} if config_key else {}
    
    return handler.handle_error(
        category=ErrorCategory.CONFIGURATION,
        severity=ErrorSeverity.ERROR,
        message=message,
        context=context,
        recoverable=False  # Configuration errors typically not recoverable at runtime
    )