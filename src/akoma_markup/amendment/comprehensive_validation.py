"""Comprehensive validation and cross-validation for amendment extraction.

Provides extensive validation rules, cross-amendment consistency checks,
and statistical validation for ensuring amendment quality and reliability.
"""

import logging
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Validation severity levels."""
    
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class ValidationRule:
    """Individual validation rule."""
    
    def __init__(
        self,
        name: str,
        check_func: callable,
        severity: Severity = Severity.WARNING,
        message: str = "",
        description: str = ""
    ):
        """Initialize validation rule.
        
        Args:
            name: Rule name.
            check_func: Function that takes amendment and returns (bool, message).
            severity: Rule severity.
            message: Rule message template.
            description: Rule description.
        """
        self.name = name
        self.check_func = check_func
        self.severity = severity
        self.message = message
        self.description = description
    
    def check(self, amendment: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Check rule against amendment.
        
        Args:
            amendment: Amendment object.
            context: Optional validation context.
            
        Returns:
            Tuple of (passed: bool, message: str).
        """
        try:
            return self.check_func(amendment, context or {})
        except Exception as e:
            logger.error(f"Rule {self.name} failed: {e}")
            return False, f"Validation error: {e}"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    
    rule_name: str
    severity: Severity
    message: str
    amendment_id: str
    amendment_details: Dict[str, Any]
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationSection:
    """Section of validation results."""
    
    name: str
    issues: List[ValidationIssue] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    
    def add_issue(
        self,
        severity: Severity,
        message: str,
        amendment: Any,
        rule_name: str = "",
        suggestions: Optional[List[str]] = None
    ) -> None:
        """Add validation issue.
        
        Args:
            severity: Issue severity.
            message: Issue message.
            amendment: Amendment object.
            rule_name: Rule name.
            suggestions: Optional suggestions.
        """
        # Create amendment identifier
        amendment_id = self._get_amendment_id(amendment)
        
        # Extract amendment details
        amendment_details = self._extract_amendment_details(amendment)
        
        issue = ValidationIssue(
            rule_name=rule_name,
            severity=severity,
            message=message,
            amendment_id=amendment_id,
            amendment_details=amendment_details,
            suggestions=suggestions or []
        )
        
        self.issues.append(issue)
        
        if severity in [Severity.ERROR, Severity.WARNING]:
            self.failed_count += 1
        else:
            self.passed_count += 1
    
    def _get_amendment_id(self, amendment: Any) -> str:
        """Get unique identifier for amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Amendment identifier.
        """
        # Try to extract key fields
        fields = self._extract_amendment_details(amendment)
        
        id_parts = []
        for field in ["section_number", "act_number", "act_year", "page_num"]:
            if field in fields:
                id_parts.append(str(fields[field]))
        
        return "_".join(id_parts) if id_parts else str(id(amendment))
    
    def _extract_amendment_details(self, amendment: Any) -> Dict[str, Any]:
        """Extract amendment details.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Dictionary of amendment details.
        """
        details = {}
        
        # Common field patterns
        field_patterns = [
            ("section_number", ["section_number", "target_section", "section"]),
            ("act_number", ["act_number", "amendment_act_id", "target_act_number"]),
            ("act_year", ["act_year", "amendment_year", "target_year"]),
            ("amendment_type", ["amendment_type", "operation"]),
            ("page_num", ["page_num", "source_page", "page"]),
            ("effective_date", ["effective_date", "date", "notification_date"]),
            ("original_text", ["original_text", "text", "content"]),
        ]
        
        for target_field, source_attrs in field_patterns:
            for source_attr in source_attrs:
                if hasattr(amendment, source_attr):
                    value = getattr(amendment, source_attr)
                    if value is not None:
                        details[target_field] = value
                        break
        
        return details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_issues": len(self.issues),
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "issues": [
                {
                    "rule_name": issue.rule_name,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "amendment_id": issue.amendment_id,
                    "amendment_details": issue.amendment_details,
                    "suggestions": issue.suggestions
                }
                for issue in self.issues
            ]
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    
    sections: List[ValidationSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(self, section: ValidationSection) -> None:
        """Add validation section.
        
        Args:
            section: Validation section.
        """
        self.sections.append(section)
    
    def generate_summary(self) -> None:
        """Generate summary statistics."""
        total_issues = sum(len(section.issues) for section in self.sections)
        total_passed = sum(section.passed_count for section in self.sections)
        total_failed = sum(section.failed_count for section in self.sections)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for section in self.sections:
            for issue in section.issues:
                severity_counts[issue.severity.value] += 1
        
        self.summary = {
            "total_issues": total_issues,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "severity_counts": dict(severity_counts),
            "success_rate": total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 1.0,
            "section_count": len(self.sections)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        self.generate_summary()
        
        return {
            "summary": self.summary,
            "sections": [section.to_dict() for section in self.sections],
            "needs_review": self.summary.get("success_rate", 1.0) < 0.8,  # <80% success rate
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results.
        
        Returns:
            List of recommendation strings.
        """
        recommendations = []
        
        # Check for common issues
        error_count = self.summary.get("severity_counts", {}).get("error", 0)
        warning_count = self.summary.get("severity_counts", {}).get("warning", 0)
        
        if error_count > 0:
            recommendations.append(f"Address {error_count} critical errors")
        
        if warning_count > 0:
            recommendations.append(f"Review {warning_count} warnings")
        
        success_rate = self.summary.get("success_rate", 1.0)
        if success_rate < 0.7:
            recommendations.append("Overall validation success rate is low - review all amendments")
        elif success_rate < 0.9:
            recommendations.append("Moderate validation success rate - review flagged amendments")
        
        return recommendations


class AmendmentValidationRules:
    """Collection of amendment validation rules."""
    
    @staticmethod
    def get_rules() -> List[ValidationRule]:
        """Get all validation rules.
        
        Returns:
            List of ValidationRule objects.
        """
        rules = []
        
        # Act/Year validation
        rules.append(ValidationRule(
            name="valid_act_year",
            check_func=AmendmentValidationRules._check_valid_act_year,
            severity=Severity.ERROR,
            message="Act year must be between 1800-2100",
            description="Validates that act year is within reasonable range"
        ))
        
        # Section number validation
        rules.append(ValidationRule(
            name="valid_section_number",
            check_func=AmendmentValidationRules._check_valid_section_number,
            severity=Severity.WARNING,
            message="Section number should be 1-999 with optional letter suffix",
            description="Validates section number format"
        ))
        
        # Date format validation
        rules.append(ValidationRule(
            name="valid_date_format",
            check_func=AmendmentValidationRules._check_valid_date_format,
            severity=Severity.WARNING,
            message="Effective date should be in valid format (DD-MM-YYYY or similar)",
            description="Validates date format"
        ))
        
        # Amendment type consistency
        rules.append(ValidationRule(
            name="type_text_consistency",
            check_func=AmendmentValidationRules._check_type_text_consistency,
            severity=Severity.WARNING,
            message="Amendment type should match text content",
            description="Checks consistency between amendment type and text"
        ))
        
        # Page boundary check
        rules.append(ValidationRule(
            name="page_within_bounds",
            check_func=AmendmentValidationRules._check_page_within_bounds,
            severity=Severity.ERROR,
            message="Page number must be within document bounds",
            description="Validates page number is within total pages"
        ))
        
        # Required fields check
        rules.append(ValidationRule(
            name="required_fields_present",
            check_func=AmendmentValidationRules._check_required_fields,
            severity=Severity.ERROR,
            message="Required fields missing: section_number and amendment_type",
            description="Checks required fields are present"
        ))
        
        # Text content check
        rules.append(ValidationRule(
            name="valid_text_content",
            check_func=AmendmentValidationRules._check_valid_text_content,
            severity=Severity.WARNING,
            message="Amendment text should be non-empty and meaningful",
            description="Validates amendment text content"
        ))
        
        # Amendment operation validity
        rules.append(ValidationRule(
            name="valid_amendment_operation",
            check_func=AmendmentValidationRules._check_valid_amendment_operation,
            severity=Severity.WARNING,
            message="Amendment operation should be valid (insert, delete, substitute, replace)",
            description="Validates amendment operation type"
        ))
        
        return rules
    
    @staticmethod
    def _check_valid_act_year(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if act year is valid.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        act_year = getattr(amendment, "act_year", None)
        if not act_year:
            return False, "Act year missing"
        
        try:
            year = int(str(act_year))
            if 1800 <= year <= 2100:
                return True, f"Act year {year} is valid"
            else:
                return False, f"Act year {year} out of range (1800-2100)"
        except ValueError:
            return False, f"Invalid act year format: {act_year}"
    
    @staticmethod
    def _check_valid_section_number(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if section number is valid.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        section = getattr(amendment, "section_number", None)
        if not section:
            # Try other field names
            section = getattr(amendment, "target_section", getattr(amendment, "section", None))
        
        if not section:
            return False, "Section number missing"
        
        section_str = str(section)
        
        # Pattern: digits optionally followed by letter
        match = re.match(r'^(\d+)([A-Z])?$', section_str)
        if not match:
            return False, f"Invalid section number format: {section_str}"
        
        # Check number range
        number_part = int(match.group(1))
        if 1 <= number_part <= 999:
            return True, f"Section number {section_str} is valid"
        else:
            return False, f"Section number {section_str} out of range (1-999)"
    
    @staticmethod
    def _check_valid_date_format(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if date format is valid.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        date_str = getattr(amendment, "effective_date", None)
        if not date_str:
            # Try other field names
            date_str = getattr(amendment, "date", getattr(amendment, "notification_date", None))
        
        if not date_str:
            return True, "No date to validate"  # Not required, so pass
        
        date_formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"
        ]
        
        for fmt in date_formats:
            try:
                datetime.strptime(str(date_str), fmt)
                return True, f"Date format valid: {date_str}"
            except ValueError:
                continue
        
        # Also accept just year
        if re.match(r'^\d{4}$', str(date_str)):
            return True, f"Year-only date: {date_str}"
        
        return False, f"Invalid date format: {date_str}"
    
    @staticmethod
    def _check_type_text_consistency(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if amendment type matches text content.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        amendment_type = getattr(amendment, "amendment_type", "").lower()
        original_text = getattr(amendment, "original_text", "")
        if not original_text:
            original_text = getattr(amendment, "text", "")
        
        if not amendment_type or not original_text:
            return True, "Insufficient data for type-text consistency check"
        
        text_lower = original_text.lower()
        
        # Check for keywords matching amendment type
        type_keywords = {
            "insert": ["insert", "add", "after", "before", "shall be inserted"],
            "delete": ["delete", "omit", "shall be omitted", "shall be deleted"],
            "substitute": ["substitute", "replace", "for the", "shall be substituted"],
            "replace": ["replace", "instead", "shall be replaced"]
        }
        
        expected_keywords = type_keywords.get(amendment_type, [])
        if expected_keywords:
            has_keyword = any(keyword in text_lower for keyword in expected_keywords)
            if has_keyword:
                return True, f"Amendment type '{amendment_type}' matches text content"
            else:
                return False, f"Amendment type '{amendment_type}' doesn't match text content"
        
        return True, "Amendment type consistency check passed"
    
    @staticmethod
    def _check_page_within_bounds(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if page number is within document bounds.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        page_num = getattr(amendment, "page_num", None)
        if not page_num:
            page_num = getattr(amendment, "source_page", None)
        
        if not page_num:
            return True, "No page number to validate"
        
        total_pages = context.get("total_pages", 0)
        if total_pages <= 0:
            return True, "No page count context available"
        
        try:
            page = int(str(page_num))
            if 1 <= page <= total_pages:
                return True, f"Page {page} within bounds (1-{total_pages})"
            else:
                return False, f"Page {page} out of bounds (1-{total_pages})"
        except ValueError:
            return False, f"Invalid page number: {page_num}"
    
    @staticmethod
    def _check_required_fields(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if required fields are present.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        missing_fields = []
        
        # Check section number
        section = getattr(amendment, "section_number", None)
        if not section:
            section = getattr(amendment, "target_section", getattr(amendment, "section", None))
        if not section:
            missing_fields.append("section_number")
        
        # Check amendment type
        amendment_type = getattr(amendment, "amendment_type", None)
        if not amendment_type:
            amendment_type = getattr(amendment, "operation", None)
        if not amendment_type:
            missing_fields.append("amendment_type")
        
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        else:
            return True, "All required fields present"
    
    @staticmethod
    def _check_valid_text_content(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if text content is valid.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        original_text = getattr(amendment, "original_text", "")
        if not original_text:
            original_text = getattr(amendment, "text", "")
        
        if not original_text:
            return False, "Amendment text is empty"
        
        text_str = str(original_text)
        
        # Check minimum length
        if len(text_str.strip()) < 10:
            return False, "Amendment text too short (minimum 10 characters)"
        
        # Check for meaningful content (not just whitespace/punctuation)
        words = text_str.split()
        if len(words) < 3:
            return False, "Amendment text has insufficient words (minimum 3)"
        
        # Check for legal amendment indicators
        legal_indicators = ["shall", "section", "act", "insert", "delete", "substitute", "replace"]
        indicator_count = sum(1 for indicator in legal_indicators if indicator.lower() in text_str.lower())
        
        if indicator_count == 0:
            return False, "Amendment text lacks legal amendment indicators"
        elif indicator_count < 2:
            return True, "Amendment text has some legal indicators"
        else:
            return True, "Amendment text has good legal indicators"
    
    @staticmethod
    def _check_valid_amendment_operation(amendment: Any, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if amendment operation is valid.
        
        Args:
            amendment: Amendment object.
            context: Validation context.
            
        Returns:
            Tuple of (passed, message).
        """
        amendment_type = getattr(amendment, "amendment_type", "").lower()
        if not amendment_type:
            amendment_type = getattr(amendment, "operation", "").lower()
        
        if not amendment_type:
            return False, "Amendment type/operation missing"
        
        valid_operations = {"insert", "delete", "substitute", "replace", "add", "omit"}
        
        if amendment_type in valid_operations:
            return True, f"Amendment operation '{amendment_type}' is valid"
        else:
            # Check if it's a close match
            for valid_op in valid_operations:
                if valid_op in amendment_type or amendment_type in valid_op:
                    return True, f"Amendment operation '{amendment_type}' is similar to valid operation '{valid_op}'"
            
            return False, f"Amendment operation '{amendment_type}' is not a standard operation"


class CrossValidator:
    """Cross-validation system for amendment consistency."""
    
    def __init__(self, enable_statistical_checks: bool = True):
        """Initialize cross-validator.
        
        Args:
            enable_statistical_checks: Whether to enable statistical checks.
        """
        self.enable_statistical_checks = enable_statistical_checks
    
    async def cross_validate(self, amendments: List[Any]) -> ValidationReport:
        """Perform cross-validation on amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            ValidationReport with cross-validation results.
        """
        report = ValidationReport()
        
        # Internal consistency checks
        report.add_section(self._validate_internal_consistency(amendments))
        
        # Cross-amendment consistency
        report.add_section(self._validate_cross_amendment_consistency(amendments))
        
        # Statistical consistency (if enabled)
        if self.enable_statistical_checks:
            report.add_section(self._validate_statistical_consistency(amendments))
        
        # Temporal consistency
        report.add_section(self._validate_temporal_consistency(amendments))
        
        # Spatial consistency (page/position)
        report.add_section(self._validate_spatial_consistency(amendments))
        
        return report
    
    def _validate_internal_consistency(self, amendments: List[Any]) -> ValidationSection:
        """Validate internal consistency of amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            ValidationSection with internal consistency results.
        """
        section = ValidationSection("Internal Consistency")
        
        # Get validation rules
        rules = AmendmentValidationRules.get_rules()
        
        # Apply each rule to each amendment
        for amendment in amendments:
            for rule in rules:
                passed, message = rule.check(amendment, {})
                
                if not passed:
                    severity = rule.severity
                    section.add_issue(
                        severity=severity,
                        message=f"{rule.name}: {message}",
                        amendment=amendment,
                        rule_name=rule.name,
                        suggestions=[f"Review {rule.description.lower()}"]
                    )
                else:
                    # Count as passed
                    section.passed_count += 1
        
        return section
    
    def _validate_cross_amendment_consistency(self, amendments: List[Any]) -> ValidationSection:
        """Validate consistency across multiple amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            ValidationSection with cross-amendment consistency results.
        """
        section = ValidationSection("Cross-Amendment Consistency")
        
        if len(amendments) <= 1:
            section.add_issue(
                severity=Severity.INFO,
                message="Only one amendment - skipping cross-amendment checks",
                amendment=amendments[0] if amendments else None
            )
            return section
        
        # Check for duplicate amendments
        duplicates = self._find_duplicate_amendments(amendments)
        if duplicates:
            for duplicate_group in duplicates:
                if len(duplicate_group) > 1:
                    for amendment in duplicate_group:
                        section.add_issue(
                            severity=Severity.WARNING,
                            message=f"Potential duplicate amendment (similar to {len(duplicate_group)-1} others)",
                            amendment=amendment,
                            rule_name="duplicate_detection",
                            suggestions=["Check if this is a true duplicate or different amendment"]
                        )
        
        # Check for conflicting amendments
        conflicts = self._find_conflicting_amendments(amendments)
        if conflicts:
            for conflict_group in conflicts:
                for amendment in conflict_group:
                    section.add_issue(
                        severity=Severity.ERROR,
                        message=f"Conflicting amendment (conflicts with {len(conflict_group)-1} others)",
                        amendment=amendment,
                        rule_name="conflict_detection",
                        suggestions=["Resolve conflicting amendment operations"]
                    )
        
        # Check amendment ordering
        ordering_issues = self._check_amendment_ordering(amendments)
        if ordering_issues:
            for issue in ordering_issues:
                section.add_issue(
                    severity=Severity.WARNING,
                    message=issue["message"],
                    amendment=issue["amendment"],
                    rule_name="ordering_check",
                    suggestions=["Review amendment sequence"]
                )
        
        return section
    
    def _find_duplicate_amendments(self, amendments: List[Any]) -> List[List[Any]]:
        """Find duplicate amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            List of duplicate groups.
        """
        # Group by key fields
        groups = defaultdict(list)
        
        for amendment in amendments:
            key = self._get_amendment_key(amendment)
            groups[key].append(amendment)
        
        # Return groups with more than one amendment
        return [group for group in groups.values() if len(group) > 1]
    
    def _get_amendment_key(self, amendment: Any) -> str:
        """Create key for amendment deduplication.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Amendment key string.
        """
        # Extract key fields
        fields = self._extract_amendment_fields(amendment)
        
        key_parts = []
        for field in ["section_number", "act_number", "act_year", "amendment_type"]:
            if field in fields:
                key_parts.append(str(fields[field]))
        
        # Also include first 50 chars of text for uniqueness
        original_text = fields.get("original_text", "")
        if original_text:
            key_parts.append(original_text[:50])
        
        return ":".join(key_parts)
    
    def _extract_amendment_fields(self, amendment: Any) -> Dict[str, Any]:
        """Extract fields from amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Dictionary of amendment fields.
        """
        fields = {}
        
        field_patterns = [
            ("section_number", ["section_number", "target_section", "section"]),
            ("act_number", ["act_number", "amendment_act_id", "target_act_number"]),
            ("act_year", ["act_year", "amendment_year", "target_year"]),
            ("amendment_type", ["amendment_type", "operation"]),
            ("original_text", ["original_text", "text", "content"]),
        ]
        
        for target_field, source_attrs in field_patterns:
            for source_attr in source_attrs:
                if hasattr(amendment, source_attr):
                    value = getattr(amendment, source_attr)
                    if value is not None:
                        fields[target_field] = value
                        break
        
        return fields
    
    def _find_conflicting_amendments(self, amendments: List[Any]) -> List[List[Any]]:
        """Find conflicting amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            List of conflicting groups.
        """
        # Group by section
        section_groups = defaultdict(list)
        
        for amendment in amendments:
            section = self._extract_section(amendment)
            if section:
                section_groups[section].append(amendment)
        
        # Check each section group for conflicts
        conflicts = []
        
        for section, section_amendments in section_groups.items():
            if len(section_amendments) > 1:
                # Check for operation conflicts
                operations = set()
                for amendment in section_amendments:
                    amendment_type = getattr(amendment, "amendment_type", "")
                    if not amendment_type:
                        amendment_type = getattr(amendment, "operation", "")
                    operations.add(amendment_type.lower())
                
                # Multiple different operations on same section is a conflict
                if len(operations) > 1:
                    conflicts.append(section_amendments)
        
        return conflicts
    
    def _extract_section(self, amendment: Any) -> Optional[str]:
        """Extract section from amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Section string or None.
        """
        section = getattr(amendment, "section_number", None)
        if not section:
            section = getattr(amendment, "target_section", getattr(amendment, "section", None))
        
        return str(section) if section else None
    
    def _check_amendment_ordering(self, amendments: List[Any]) -> List[Dict[str, Any]]:
        """Check amendment ordering.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            List of ordering issues.
        """
        issues = []
        
        # Check page number ordering
        page_numbers = []
        for i, amendment in enumerate(amendments):
            page_num = getattr(amendment, "page_num", None)
            if page_num:
                try:
                    page_numbers.append((i, int(str(page_num))))
                except ValueError:
                    pass
        
        # Sort by actual order in list
        actual_order = [pn for _, pn in page_numbers]
        sorted_order = sorted(actual_order)
        
        if actual_order != sorted_order:
            for i, (amendment_index, page_num) in enumerate(page_numbers):
                if i < len(sorted_order) and page_num != sorted_order[i]:
                    issues.append({
                        "amendment": amendments[amendment_index],
                        "message": f"Page number {page_num} out of sequence"
                    })
        
        # Check section number ordering (if available)
        section_numbers = []
        for i, amendment in enumerate(amendments):
            section = self._extract_section(amendment)
            if section:
                # Extract numeric part
                match = re.search(r'(\d+)', section)
                if match:
                    try:
                        section_num = int(match.group(1))
                        section_numbers.append((i, section_num, section))
                    except ValueError:
                        pass
        
        if len(section_numbers) > 1:
            # Check if sections are in increasing order
            for i in range(1, len(section_numbers)):
                prev_index, prev_num, prev_section = section_numbers[i-1]
                curr_index, curr_num, curr_section = section_numbers[i]
                
                if curr_num < prev_num:
                    issues.append({
                        "amendment": amendments[curr_index],
                        "message": f"Section {curr_section} appears before section {prev_section}"
                    })
        
        return issues
    
    def _validate_statistical_consistency(self, amendments: List[Any]) -> ValidationSection:
        """Validate statistical consistency of amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            ValidationSection with statistical consistency results.
        """
        section = ValidationSection("Statistical Consistency")
        
        if len(amendments) < 3:
            section.add_issue(
                severity=Severity.INFO,
                message="Insufficient amendments for statistical analysis",
                amendment=amendments[0] if amendments else None
            )
            return section
        
        # Collect statistics
        page_numbers = []
        section_numbers = []
        
        for amendment in amendments:
            # Page numbers
            page_num = getattr(amendment, "page_num", None)
            if page_num:
                try:
                    page_numbers.append(int(str(page_num)))
                except ValueError:
                    pass
            
            # Section numbers (numeric part)
            section = self._extract_section(amendment)
            if section:
                match = re.search(r'(\d+)', section)
                if match:
                    try:
                        section_numbers.append(int(match.group(1)))
                    except ValueError:
                        pass
        
        # Analyze page number distribution
        if len(page_numbers) >= 3:
            page_mean = statistics.mean(page_numbers)
            page_stdev = statistics.stdev(page_numbers) if len(page_numbers) > 1 else 0
            
            # Check for outliers (more than 2 standard deviations from mean)
            for i, page_num in enumerate(page_numbers):
                if page_stdev > 0:
                    z_score = abs((page_num - page_mean) / page_stdev)
                    if z_score > 2.0:
                        section.add_issue(
                            severity=Severity.WARNING,
                            message=f"Page number {page_num} is statistical outlier (z-score: {z_score:.2f})",
                            amendment=amendments[i],
                            rule_name="statistical_outlier",
                            suggestions=["Verify page number accuracy"]
                        )
        
        # Analyze section number distribution
        if len(section_numbers) >= 3:
            section_mean = statistics.mean(section_numbers)
            section_stdev = statistics.stdev(section_numbers) if len(section_numbers) > 1 else 0
            
            # Check for gaps in section numbers
            sorted_sections = sorted(set(section_numbers))
            if len(sorted_sections) > 1:
                gaps = []
                for i in range(1, len(sorted_sections)):
                    gap = sorted_sections[i] - sorted_sections[i-1]
                    if gap > 10:  # Large gap
                        gaps.append((sorted_sections[i-1], sorted_sections[i], gap))
                
                for start, end, gap in gaps:
                    section.add_issue(
                        severity=Severity.INFO,
                        message=f"Large gap in section numbers: {start} to {end} (gap: {gap})",
                        amendment=amendments[0],  # Reference amendment
                        rule_name="section_gap",
                        suggestions=[f"Check for missing amendments between sections {start} and {end}"]
                    )
        
        return section
    
    def _validate_temporal_consistency(self, amendments: List[Any]) -> ValidationSection:
        """Validate temporal consistency of amendments.
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            ValidationSection with temporal consistency results.
        """
        section = ValidationSection("Temporal Consistency")
        
        # Extract dates
        dates = []
        for i, amendment in enumerate(amendments):
            date_str = getattr(amendment, "effective_date", None)
            if not date_str:
                date_str = getattr(amendment, "date", None)
            
            if date_str:
                dates.append((i, str(date_str)))
        
        if len(dates) < 2:
            section.add_issue(
                severity=Severity.INFO,
                message="Insufficient dates for temporal analysis",
                amendment=amendments[0] if amendments else None
            )
            return section
        
        # Try to parse dates
        parsed_dates = []
        date_formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"
        ]
        
        for i, date_str in dates:
            parsed = None
            for fmt in date_formats:
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if parsed:
                parsed_dates.append((i, parsed))
            else:
                # Try year-only
                match = re.search(r'\b(\d{4})\b', date_str)
                if match:
                    try:
                        parsed = datetime.strptime(f"{match.group(1)}-01-01", "%Y-%m-%d")
                        parsed_dates.append((i, parsed))
                    except ValueError:
                        pass
        
        if len(parsed_dates) >= 2:
            # Check chronological order
            sorted_by_date = sorted(parsed_dates, key=lambda x: x[1])
            
            # Compare with original order
            if parsed_dates != sorted_by_date:
                for (i, date_obj) in parsed_dates:
                    section.add_issue(
                        severity=Severity.WARNING,
                        message="Amendment dates not in chronological order",
                        amendment=amendments[i],
                        rule_name="temporal_order",
                        suggestions=["Review amendment date sequence"]
                    )
            
            # Check date range
            dates_only = [date_obj for _, date_obj in parsed_dates]
            min_date = min(dates_only)
            max_date = max(dates_only)
            date_range = (max_date - min_date).days
            
            if date_range > 365:  # More than 1 year
                section.add_issue(
                    severity=Severity.INFO,
                    message=f"Amendment dates span {date_range} days (more than 1 year)",
                    amendment=amendments[parsed_dates[0][0]],
                    rule_name="date_range",
                    suggestions=["Verify all amendments belong to same legislative session"]
                )
        
        return section
    
    def _validate_spatial_consistency(self, amendments: List[Any]) -> ValidationSection:
        """Validate spatial consistency (page/position).
        
        Args:
            amendments: List of amendment objects.
            
        Returns:
            ValidationSection with spatial consistency results.
        """
        section = ValidationSection("Spatial Consistency")
        
        # Group by page
        page_groups = defaultdict(list)
        for i, amendment in enumerate(amendments):
            page_num = getattr(amendment, "page_num", None)
            if page_num:
                try:
                    page = int(str(page_num))
                    page_groups[page].append((i, amendment))
                except ValueError:
                    pass
        
        # Check each page group
        for page, page_amendments in page_groups.items():
            if len(page_amendments) > 5:  # Many amendments on same page
                for i, amendment in page_amendments:
                    section.add_issue(
                        severity=Severity.INFO,
                        message=f"Page {page} has {len(page_amendments)} amendments (many amendments on single page)",
                        amendment=amendment,
                        rule_name="page_density",
                        suggestions=["Verify amendments are correctly assigned to pages"]
                    )
        
        # Check page distribution
        if page_groups:
            pages = list(page_groups.keys())
            page_range = max(pages) - min(pages) + 1
            
            if page_range > 10 and len(amendments) < page_range / 2:
                # Sparse distribution
                section.add_issue(
                    severity=Severity.INFO,
                    message=f"Amendments spread across {page_range} pages but only {len(amendments)} amendments",
                    amendment=amendments[0],
                    rule_name="sparse_distribution",
                    suggestions=["Check for missing amendments in page range"]
                )
        
        return section


def validate_amendments_comprehensive(
    amendments: List[Any],
    total_pages: int = 0,
    enable_cross_validation: bool = True
) -> Dict[str, Any]:
    """Comprehensive validation of amendments.
    
    Args:
        amendments: List of amendment objects.
        total_pages: Total pages in document.
        enable_cross_validation: Whether to enable cross-validation.
        
    Returns:
        Dictionary with validation results.
    """
    logger.info(f"Starting comprehensive validation of {len(amendments)} amendments")
    
    # Create context
    context = {"total_pages": total_pages}
    
    # Create validator
    validator = CrossValidator(enable_statistical_checks=enable_cross_validation)
    
    # Run validation
    report = validator.cross_validate(amendments)
    
    # Also apply individual rules
    rules_section = ValidationSection("Individual Rule Validation")
    rules = AmendmentValidationRules.get_rules()
    
    for amendment in amendments:
        for rule in rules:
            passed, message = rule.check(amendment, context)
            
            if not passed:
                rules_section.add_issue(
                    severity=rule.severity,
                    message=f"{rule.name}: {message}",
                    amendment=amendment,
                    rule_name=rule.name
                )
            else:
                rules_section.passed_count += 1
    
    report.add_section(rules_section)
    
    # Generate final report
    result = report.to_dict()
    
    logger.info(f"Validation complete: {result['summary'].get('success_rate', 0):.1%} success rate")
    
    return result