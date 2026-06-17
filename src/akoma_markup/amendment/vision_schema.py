"""Vision-based amendment extraction schema.

Defines the data structures for amendments extracted via multimodal LLM analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict, List
from datetime import datetime


@dataclass
class VisionExtractedAmendment:
    """Amendment extracted through multimodal LLM analysis of PDF pages."""
    
    # Basic identification
    page_num: int
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) coordinates on page
    amendment_type: str  # "substitution", "insertion", "omission", "repeal", etc.
    
    # Act information
    act_number: str
    act_year: str
    
    # Section information
    section_number: str
    target_section: str  # The section being amended
    target_location: str  # "subsection (4)", "clause (a)", etc.
    
    # Text content
    original_text: str  # Full text being replaced/inserted/deleted
    effective_date: str
    new_text: Optional[str] = None  # For substitutions, the new text
    
    # Metadata
    footnote_marker: str = ""
    confidence_score: float = 1.0
    visual_context: str = ""  # Description of surrounding page context
    
    # Raw data
    raw_llm_response: Dict[str, Any] = field(default_factory=dict)
    
    # Additional metadata
    extracted_at: datetime = field(default_factory=datetime.now)
    source_pdf: Optional[str] = None
    gazette_notification: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        
        # Handle datetime serialization
        result['extracted_at'] = self.extracted_at.isoformat()
        
        # Handle tuple serialization
        result['bbox'] = list(self.bbox)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisionExtractedAmendment:
        """Create instance from dictionary (JSON deserialization)."""
        # Handle datetime deserialization
        if 'extracted_at' in data and isinstance(data['extracted_at'], str):
            data['extracted_at'] = datetime.fromisoformat(data['extracted_at'])
        
        # Handle tuple deserialization
        if 'bbox' in data and isinstance(data['bbox'], list):
            data['bbox'] = tuple(data['bbox'])
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> VisionExtractedAmendment:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate the amendment data and return list of errors."""
        errors = []
        
        # Required fields
        if self.page_num < 1:
            errors.append("page_num must be positive")
        
        if not self.act_number:
            errors.append("act_number is required")
        
        if not self.act_year:
            errors.append("act_year is required")
        
        if not self.section_number:
            errors.append("section_number is required")
        
        if not self.amendment_type:
            errors.append("amendment_type is required")
        elif self.amendment_type not in ["substitution", "insertion", "omission", "repeal", "addition"]:
            errors.append(f"Invalid amendment_type: {self.amendment_type}")
        
        if not self.original_text:
            errors.append("original_text is required")
        
        # Bbox validation
        if len(self.bbox) != 4:
            errors.append("bbox must be a 4-tuple")
        elif self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            errors.append("bbox coordinates must be valid (x0 < x1, y0 < y1)")
        
        # Confidence score validation
        if not 0.0 <= self.confidence_score <= 1.0:
            errors.append("confidence_score must be between 0.0 and 1.0")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the amendment is valid."""
        return len(self.validate()) == 0
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (f"Amendment to {self.act_number} ({self.act_year}) "
                f"Section {self.section_number} - {self.amendment_type} "
                f"(page {self.page_num}, confidence: {self.confidence_score:.2f})")


@dataclass
class AmendmentExtractionResult:
    """Result of amendment extraction from a PDF."""
    
    pdf_path: str
    extracted_amendments: List[VisionExtractedAmendment]
    total_pages_processed: int
    extraction_date: datetime = field(default_factory=datetime.now)
    extraction_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'pdf_path': self.pdf_path,
            'extracted_amendments': [am.to_dict() for am in self.extracted_amendments],
            'total_pages_processed': self.total_pages_processed,
            'extraction_date': self.extraction_date.isoformat(),
            'extraction_errors': self.extraction_errors,
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AmendmentExtractionResult:
        """Create instance from dictionary."""
        data['extraction_date'] = datetime.fromisoformat(data['extraction_date'])
        data['extracted_amendments'] = [
            VisionExtractedAmendment.from_dict(am) 
            for am in data['extracted_amendments']
        ]
        return cls(**data)