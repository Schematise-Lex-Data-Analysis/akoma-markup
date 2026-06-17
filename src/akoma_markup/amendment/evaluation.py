"""Evaluation metrics for amendment extraction methods.

Compares vision-based extraction with regex-based extraction
and calculates performance metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Evaluation report comparing extraction methods."""
    
    pdf_path: str
    evaluation_date: str
    vision_amendments_count: int
    regex_amendments_count: int
    common_amendments_count: int
    vision_only_count: int
    regex_only_count: int
    
    # Precision/Recall metrics (if ground truth available)
    vision_precision: Optional[float] = None
    vision_recall: Optional[float] = None
    vision_f1: Optional[float] = None
    
    regex_precision: Optional[float] = None
    regex_recall: Optional[float] = None
    regex_f1: Optional[float] = None
    
    # Comparison metrics
    agreement_rate: Optional[float] = None
    jaccard_similarity: Optional[float] = None
    
    # Performance metrics
    vision_processing_time: Optional[float] = None  # seconds
    regex_processing_time: Optional[float] = None  # seconds
    
    # Detailed comparison
    comparison_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pdf_path": self.pdf_path,
            "evaluation_date": self.evaluation_date,
            "counts": {
                "vision": self.vision_amendments_count,
                "regex": self.regex_amendments_count,
                "common": self.common_amendments_count,
                "vision_only": self.vision_only_count,
                "regex_only": self.regex_only_count,
            },
            "metrics": {
                "vision": {
                    "precision": self.vision_precision,
                    "recall": self.vision_recall,
                    "f1": self.vision_f1,
                },
                "regex": {
                    "precision": self.regex_precision,
                    "recall": self.regex_recall,
                    "f1": self.regex_f1,
                },
                "comparison": {
                    "agreement_rate": self.agreement_rate,
                    "jaccard_similarity": self.jaccard_similarity,
                }
            },
            "performance": {
                "vision_processing_time": self.vision_processing_time,
                "regex_processing_time": self.regex_processing_time,
            },
            "comparison_details": self.comparison_details,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Amendment Extraction Evaluation Report",
            "",
            f"**PDF**: {self.pdf_path}",
            f"**Evaluation Date**: {self.evaluation_date}",
            "",
            "## Summary",
            "",
            f"- **Vision Amendments**: {self.vision_amendments_count}",
            f"- **Regex Amendments**: {self.regex_amendments_count}",
            f"- **Common Amendments**: {self.common_amendments_count}",
            f"- **Vision Only**: {self.vision_only_count}",
            f"- **Regex Only**: {self.regex_only_count}",
            "",
            "## Performance Metrics",
            "",
        ]
        
        if self.vision_processing_time is not None:
            lines.append(f"- **Vision Processing Time**: {self.vision_processing_time:.2f} seconds")
        
        if self.regex_processing_time is not None:
            lines.append(f"- **Regex Processing Time**: {self.regex_processing_time:.2f} seconds")
        
        if self.agreement_rate is not None:
            lines.append(f"- **Agreement Rate**: {self.agreement_rate:.1%}")
        
        if self.jaccard_similarity is not None:
            lines.append(f"- **Jaccard Similarity**: {self.jaccard_similarity:.1%}")
        
        # Add precision/recall if available
        if self.vision_precision is not None:
            lines.extend([
                "",
                "## Accuracy Metrics (vs Ground Truth)",
                "",
                "### Vision Extraction",
                f"- **Precision**: {self.vision_precision:.1%}",
                f"- **Recall**: {self.vision_recall:.1%}",
                f"- **F1 Score**: {self.vision_f1:.1%}",
                "",
                "### Regex Extraction",
                f"- **Precision**: {self.regex_precision:.1%}",
                f"- **Recall**: {self.regex_recall:.1%}",
                f"- **F1 Score**: {self.regex_f1:.1%}",
            ])
        
        return "\n".join(lines)


class ExtractionEvaluator:
    """Evaluator for comparing amendment extraction methods."""
    
    def compare_vision_vs_regex(
        self, 
        pdf_path: Path,
        vision_amendments: List[Any],
        regex_amendments: List[Any],
        vision_processing_time: Optional[float] = None,
        regex_processing_time: Optional[float] = None
    ) -> EvaluationReport:
        """Compare vision and regex extraction results.
        
        Args:
            pdf_path: Path to PDF file.
            vision_amendments: Amendments from vision extraction.
            regex_amendments: Amendments from regex extraction.
            vision_processing_time: Optional vision processing time in seconds.
            regex_processing_time: Optional regex processing time in seconds.
            
        Returns:
            EvaluationReport with comparison metrics.
        """
        logger.info(f"Comparing extraction methods for {pdf_path}")
        
        # Convert to comparable format
        vision_keys = self._extract_amendment_keys(vision_amendments, "vision")
        regex_keys = self._extract_amendment_keys(regex_amendments, "regex")
        
        # Calculate counts
        vision_count = len(vision_amendments)
        regex_count = len(regex_amendments)
        
        vision_key_set = set(vision_keys.keys())
        regex_key_set = set(regex_keys.keys())
        
        common_keys = vision_key_set.intersection(regex_key_set)
        vision_only_keys = vision_key_set - regex_key_set
        regex_only_keys = regex_key_set - vision_key_set
        
        common_count = len(common_keys)
        vision_only_count = len(vision_only_keys)
        regex_only_count = len(regex_only_keys)
        
        # Calculate agreement metrics
        agreement_rate = common_count / max(vision_count, regex_count) if max(vision_count, regex_count) > 0 else 0
        jaccard_similarity = common_count / len(vision_key_set.union(regex_key_set)) if len(vision_key_set.union(regex_key_set)) > 0 else 0
        
        # Create comparison details
        comparison_details = {
            "common_amendments": [
                self._create_comparison_entry(vision_keys[key], regex_keys[key])
                for key in common_keys
            ],
            "vision_only_amendments": [
                self._amendment_to_dict(vision_keys[key], "vision")
                for key in vision_only_keys
            ],
            "regex_only_amendments": [
                self._amendment_to_dict(regex_keys[key], "regex")
                for key in regex_only_keys
            ],
        }
        
        return EvaluationReport(
            pdf_path=str(pdf_path),
            evaluation_date=datetime.now().isoformat(),
            vision_amendments_count=vision_count,
            regex_amendments_count=regex_count,
            common_amendments_count=common_count,
            vision_only_count=vision_only_count,
            regex_only_count=regex_only_count,
            agreement_rate=agreement_rate,
            jaccard_similarity=jaccard_similarity,
            vision_processing_time=vision_processing_time,
            regex_processing_time=regex_processing_time,
            comparison_details=comparison_details,
        )
    
    def calculate_precision_recall(
        self, 
        ground_truth: List[Any], 
        extracted: List[Any],
        method_name: str = "extraction"
    ) -> Dict[str, Any]:
        """Calculate precision, recall, and F1 score.
        
        Args:
            ground_truth: Ground truth amendments.
            extracted: Extracted amendments.
            method_name: Name of the extraction method for logging.
            
        Returns:
            Dictionary with precision, recall, and F1 score.
        """
        logger.info(f"Calculating precision/recall for {method_name}")
        
        # Convert to comparable format
        truth_keys = self._extract_amendment_keys(ground_truth, "truth")
        extracted_keys = self._extract_amendment_keys(extracted, "extracted")
        
        truth_key_set = set(truth_keys.keys())
        extracted_key_set = set(extracted_keys.keys())
        
        # True positives: amendments correctly extracted
        true_positives = truth_key_set.intersection(extracted_key_set)
        
        # False positives: amendments extracted but not in ground truth
        false_positives = extracted_key_set - truth_key_set
        
        # False negatives: amendments in ground truth but not extracted
        false_negatives = truth_key_set - extracted_key_set
        
        # Calculate metrics
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Detailed analysis
        detailed_false_positives = [
            self._amendment_to_dict(extracted_keys[key], "false_positive")
            for key in false_positives
        ]
        
        detailed_false_negatives = [
            self._amendment_to_dict(truth_keys[key], "false_negative")
            for key in false_negatives
        ]
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "counts": {
                "true_positives": tp_count,
                "false_positives": fp_count,
                "false_negatives": fn_count,
                "ground_truth_total": len(ground_truth),
                "extracted_total": len(extracted),
            },
            "details": {
                "false_positives": detailed_false_positives,
                "false_negatives": detailed_false_negatives,
            }
        }
    
    def generate_confusion_matrix(
        self, 
        vision_results: List[Any], 
        regex_results: List[Any]
    ) -> Dict[str, Any]:
        """Generate confusion matrix comparing vision and regex results.
        
        Args:
            vision_results: Amendments from vision extraction.
            regex_results: Amendments from regex extraction.
            
        Returns:
            Dictionary representing confusion matrix.
        """
        vision_keys = self._extract_amendment_keys(vision_results, "vision")
        regex_keys = self._extract_amendment_keys(regex_results, "regex")
        
        vision_key_set = set(vision_keys.keys())
        regex_key_set = set(regex_keys.keys())
        
        # Create matrix
        matrix = {
            "vision_only": len(vision_key_set - regex_key_set),
            "regex_only": len(regex_key_set - vision_key_set),
            "both": len(vision_key_set.intersection(regex_key_set)),
            "total_vision": len(vision_results),
            "total_regex": len(regex_results),
        }
        
        # Calculate percentages
        total_unique = len(vision_key_set.union(regex_key_set))
        if total_unique > 0:
            matrix["vision_only_percent"] = matrix["vision_only"] / total_unique
            matrix["regex_only_percent"] = matrix["regex_only"] / total_unique
            matrix["both_percent"] = matrix["both"] / total_unique
        
        return matrix
    
    def _extract_amendment_keys(self, amendments: List[Any], source: str) -> Dict[str, Any]:
        """Extract unique keys for amendments.
        
        Args:
            amendments: List of amendments.
            source: Source identifier.
            
        Returns:
            Dictionary mapping keys to amendments.
        """
        keys = {}
        
        for i, amendment in enumerate(amendments):
            key = self._create_amendment_key(amendment, source, i)
            if key:
                keys[key] = amendment
        
        return keys
    
    def _create_amendment_key(self, amendment: Any, source: str, index: int) -> Optional[str]:
        """Create unique key for an amendment.
        
        Args:
            amendment: Amendment object.
            source: Source identifier.
            index: Amendment index.
            
        Returns:
            Unique key string or None.
        """
        try:
            # Try to extract common fields
            fields = []
            
            # Page number
            if hasattr(amendment, "page_num"):
                fields.append(f"page:{amendment.page_num}")
            elif hasattr(amendment, "source_page"):
                fields.append(f"page:{amendment.source_page}")
            
            # Section/target
            if hasattr(amendment, "section_number"):
                fields.append(f"section:{amendment.section_number}")
            elif hasattr(amendment, "target_section"):
                fields.append(f"section:{amendment.target_section}")
            elif hasattr(amendment, "section"):
                fields.append(f"section:{amendment.section}")
            
            # Amendment type
            if hasattr(amendment, "amendment_type"):
                fields.append(f"type:{amendment.amendment_type}")
            
            # Text snippet (first 50 chars)
            if hasattr(amendment, "original_text") and amendment.original_text:
                fields.append(f"text:{amendment.original_text[:50]}")
            elif hasattr(amendment, "text") and amendment.text:
                fields.append(f"text:{amendment.text[:50]}")
            
            # If we have enough fields, create key
            if len(fields) >= 2:
                return f"{source}:{':'.join(fields)}"
            else:
                # Fallback: use index
                return f"{source}:index:{index}"
                
        except Exception as e:
            logger.warning(f"Failed to create key for amendment: {e}")
            return f"{source}:error:{index}"
    
    def _create_comparison_entry(self, vision_am: Any, regex_am: Any) -> Dict[str, Any]:
        """Create comparison entry for common amendments.
        
        Args:
            vision_am: Vision amendment.
            regex_am: Regex amendment.
            
        Returns:
            Dictionary with comparison details.
        """
        entry = {
            "vision": self._amendment_to_dict(vision_am, "vision"),
            "regex": self._amendment_to_dict(regex_am, "regex"),
            "comparison": {},
        }
        
        # Compare confidence if available
        vision_conf = getattr(vision_am, "confidence_score", None)
        regex_conf = getattr(regex_am, "linkage_confidence", None)
        
        if vision_conf is not None or regex_conf is not None:
            entry["comparison"]["confidence"] = {
                "vision": vision_conf,
                "regex": regex_conf,
            }
        
        return entry
    
    def _amendment_to_dict(self, amendment: Any, source: str) -> Dict[str, Any]:
        """Convert amendment to dictionary for serialization.
        
        Args:
            amendment: Amendment object.
            source: Source identifier.
            
        Returns:
            Dictionary representation.
        """
        result = {"source": source}
        
        # Try common fields
        common_fields = [
            "page_num", "source_page", "section_number", "target_section",
            "section", "amendment_type", "amendment_act_id", "act_number",
            "act_year", "original_text", "text", "new_text", "effective_date",
            "confidence_score", "linkage_confidence", "target_location",
        ]
        
        for field in common_fields:
            if hasattr(amendment, field):
                value = getattr(amendment, field)
                if value is not None:
                    result[field] = value
        
        # If amendment has to_dict method, use it
        if hasattr(amendment, "to_dict"):
            try:
                result.update(amendment.to_dict())
            except Exception:
                pass
        
        return result


# Convenience functions
def evaluate_extraction_methods(
    pdf_path: Path,
    vision_amendments: List[Any],
    regex_amendments: List[Any],
    output_path: Optional[Path] = None
) -> EvaluationReport:
    """Convenience function for evaluating extraction methods.
    
    Args:
        pdf_path: Path to PDF file.
        vision_amendments: Amendments from vision extraction.
        regex_amendments: Amendments from regex extraction.
        output_path: Optional path to save report.
        
    Returns:
        EvaluationReport.
    """
    evaluator = ExtractionEvaluator()
    report = evaluator.compare_vision_vs_regex(pdf_path, vision_amendments, regex_amendments)
    
    if output_path:
        # Save report
        if output_path.suffix.lower() == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report.to_json())
        else:
            # Default to markdown
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
        
        logger.info(f"Evaluation report saved to: {output_path}")
    
    return report