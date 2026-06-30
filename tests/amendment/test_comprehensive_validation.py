"""Tests for comprehensive_validation module."""

import unittest
from unittest.mock import Mock
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.akoma_markup.amendment.comprehensive_validation import (
    Severity,
    ValidationRule,
    ValidationIssue,
    ValidationSection,
    ValidationReport,
    CrossValidator
)


class TestSeverity(unittest.TestCase):
    """Test Severity enum."""
    
    def test_severity_values(self):
        """Test Severity enum values."""
        self.assertEqual(Severity.ERROR.value, "error")
        self.assertEqual(Severity.WARNING.value, "warning")
        self.assertEqual(Severity.INFO.value, "info")
        self.assertEqual(Severity.SUCCESS.value, "success")
    
    def test_severity_from_string(self):
        """Test creating Severity from string."""
        self.assertEqual(Severity("error"), Severity.ERROR)
        self.assertEqual(Severity("warning"), Severity.WARNING)
        self.assertEqual(Severity("info"), Severity.INFO)
        self.assertEqual(Severity("success"), Severity.SUCCESS)


class TestValidationRule(unittest.TestCase):
    """Test ValidationRule class."""
    
    def test_validation_rule_init(self):
        """Test ValidationRule initialization."""
        def check_func(amendment, context):
            return True, "All good"
        
        rule = ValidationRule(
            name="test_rule",
            check_func=check_func,
            severity=Severity.WARNING,
            message="Test message",
            description="Test description"
        )
        
        self.assertEqual(rule.name, "test_rule")
        self.assertEqual(rule.check_func, check_func)
        self.assertEqual(rule.severity, Severity.WARNING)
        self.assertEqual(rule.message, "Test message")
        self.assertEqual(rule.description, "Test description")
    
    def test_validation_rule_check_success(self):
        """Test ValidationRule check method success."""
        def check_func(amendment, context):
            return True, "All good"
        
        rule = ValidationRule(
            name="test_rule",
            check_func=check_func,
            severity=Severity.WARNING
        )
        
        amendment = Mock()
        context = {}
        
        passed, message = rule.check(amendment, context)
        
        self.assertTrue(passed)
        self.assertEqual(message, "All good")
    
    def test_validation_rule_check_failure(self):
        """Test ValidationRule check method failure."""
        def check_func(amendment, context):
            return False, "Failed check"
        
        rule = ValidationRule(
            name="test_rule",
            check_func=check_func,
            severity=Severity.ERROR
        )
        
        amendment = Mock()
        context = {}
        
        passed, message = rule.check(amendment, context)
        
        self.assertFalse(passed)
        self.assertEqual(message, "Failed check")
    
    def test_validation_rule_check_exception(self):
        """Test ValidationRule check method with exception."""
        def check_func(amendment, context):
            raise ValueError("Test error")
        
        rule = ValidationRule(
            name="test_rule",
            check_func=check_func,
            severity=Severity.ERROR
        )
        
        amendment = Mock()
        context = {}
        
        passed, message = rule.check(amendment, context)
        
        self.assertFalse(passed)
        self.assertIn("Validation error:", message)


class TestValidationIssue(unittest.TestCase):
    """Test ValidationIssue class."""
    
    def test_validation_issue_init(self):
        """Test ValidationIssue initialization."""
        amendment_details = {"section_number": "45A", "operation": "insert"}
        suggestions = ["Check section reference", "Verify with gazette"]
        
        issue = ValidationIssue(
            rule_name="section_number_check",
            severity=Severity.WARNING,
            message="Section number may be incorrect",
            amendment_id="amd_123",
            amendment_details=amendment_details,
            suggestions=suggestions
        )
        
        self.assertEqual(issue.rule_name, "section_number_check")
        self.assertEqual(issue.severity, Severity.WARNING)
        self.assertEqual(issue.message, "Section number may be incorrect")
        self.assertEqual(issue.amendment_id, "amd_123")
        self.assertEqual(issue.amendment_details, amendment_details)
        self.assertEqual(issue.suggestions, suggestions)
    
    def test_validation_issue_to_dict(self):
        """Test ValidationIssue to_dict method."""
        issue = ValidationIssue(
            rule_name="test_rule",
            severity=Severity.ERROR,
            message="Test error",
            amendment_id="amd_123",
            amendment_details={"test": "data"}
        )
        
        result = issue.to_dict()
        
        self.assertEqual(result["rule_name"], "test_rule")
        self.assertEqual(result["severity"], "error")
        self.assertEqual(result["message"], "Test error")
        self.assertEqual(result["amendment_id"], "amd_123")
        self.assertEqual(result["amendment_details"], {"test": "data"})


class TestValidationSection(unittest.TestCase):
    """Test ValidationSection class."""
    
    def test_validation_section_init(self):
        """Test ValidationSection initialization."""
        issues = [
            ValidationIssue(
                rule_name="rule1",
                severity=Severity.WARNING,
                message="Warning 1",
                amendment_id="amd1",
                amendment_details={}
            ),
            ValidationIssue(
                rule_name="rule2",
                severity=Severity.ERROR,
                message="Error 1",
                amendment_id="amd2",
                amendment_details={}
            )
        ]
        
        section = ValidationSection(
            name="section_validation",
            issues=issues,
            passed_count=5,
            failed_count=2
        )
        
        self.assertEqual(section.name, "section_validation")
        self.assertEqual(len(section.issues), 2)
        self.assertEqual(section.passed_count, 5)
        self.assertEqual(section.failed_count, 2)
    
    def test_add_issue(self):
        """Test adding issue to ValidationSection."""
        section = ValidationSection(name="test_section")
        
        amendment = Mock()
        amendment.amendment_id = "amd_123"
        
        section.add_issue(
            severity=Severity.WARNING,
            message="Test warning",
            amendment=amendment,
            rule_name="test_rule",
            suggestions=["Suggestion 1", "Suggestion 2"]
        )
        
        self.assertEqual(len(section.issues), 1)
        self.assertEqual(section.failed_count, 1)
        
        issue = section.issues[0]
        self.assertEqual(issue.rule_name, "test_rule")
        self.assertEqual(issue.severity, Severity.WARNING)
        self.assertEqual(issue.message, "Test warning")
        self.assertEqual(issue.amendment_id, "amd_123")
        self.assertEqual(issue.suggestions, ["Suggestion 1", "Suggestion 2"])
    
    def test_to_dict(self):
        """Test ValidationSection to_dict method."""
        issue = ValidationIssue(
            rule_name="test_rule",
            severity=Severity.WARNING,
            message="Test message",
            amendment_id="amd_123",
            amendment_details={}
        )
        
        section = ValidationSection(
            name="test_section",
            issues=[issue],
            passed_count=3,
            failed_count=1
        )
        
        result = section.to_dict()
        
        self.assertEqual(result["name"], "test_section")
        self.assertEqual(result["passed_count"], 3)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["total_issues"], 1)
        self.assertEqual(len(result["issues"]), 1)
        self.assertEqual(result["issues"][0]["rule_name"], "test_rule")


class TestValidationReport(unittest.TestCase):
    """Test ValidationReport class."""
    
    def test_validation_report_init(self):
        """Test ValidationReport initialization."""
        section = ValidationSection(
            name="test_section",
            passed_count=5,
            failed_count=2
        )
        
        report = ValidationReport(
            amendment_count=10,
            sections={"test_section": section},
            overall_score=0.8,
            needs_review=True
        )
        
        self.assertEqual(report.amendment_count, 10)
        self.assertEqual(len(report.sections), 1)
        self.assertEqual(report.overall_score, 0.8)
        self.assertTrue(report.needs_review)
        self.assertEqual(report.total_issues, 2)  # failed_count from section
    
    def test_add_section(self):
        """Test adding section to ValidationReport."""
        report = ValidationReport(amendment_count=5)
        
        section = ValidationSection(
            name="new_section",
            passed_count=3,
            failed_count=1
        )
        
        report.add_section(section)
        
        self.assertIn("new_section", report.sections)
        self.assertEqual(report.total_issues, 1)
    
    def test_get_issues_by_severity(self):
        """Test getting issues by severity."""
        error_issue = ValidationIssue(
            rule_name="error_rule",
            severity=Severity.ERROR,
            message="Error",
            amendment_id="amd1",
            amendment_details={}
        )
        
        warning_issue = ValidationIssue(
            rule_name="warning_rule",
            severity=Severity.WARNING,
            message="Warning",
            amendment_id="amd2",
            amendment_details={}
        )
        
        section = ValidationSection(
            name="test_section",
            issues=[error_issue, warning_issue]
        )
        
        report = ValidationReport(amendment_count=2)
        report.add_section(section)
        
        errors = report.get_issues_by_severity(Severity.ERROR)
        warnings = report.get_issues_by_severity(Severity.WARNING)
        
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].rule_name, "error_rule")
        
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].rule_name, "warning_rule")
    
    def test_to_dict(self):
        """Test ValidationReport to_dict method."""
        section = ValidationSection(
            name="test_section",
            passed_count=3,
            failed_count=1
        )
        
        report = ValidationReport(
            amendment_count=5,
            sections={"test_section": section},
            overall_score=0.75,
            needs_review=False
        )
        
        result = report.to_dict()
        
        self.assertEqual(result["amendment_count"], 5)
        self.assertEqual(result["overall_score"], 0.75)
        self.assertEqual(result["needs_review"], False)
        self.assertEqual(result["total_issues"], 1)
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"]["test_section"]["name"], "test_section")


class TestCrossValidator(unittest.TestCase):
    """Test CrossValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = CrossValidator()
    
    def test_init(self):
        """Test ComprehensiveValidator initialization."""
        self.assertGreater(len(self.validator.rules), 0)
        self.assertIn("section_number_format", self.validator.rules)
        self.assertIn("operation_validity", self.validator.rules)
        self.assertIn("cross_amendment_consistency", self.validator.rules)
    
    def test_register_rule(self):
        """Test registering a custom rule."""
        def custom_check(amendment, context):
            return True, "Custom check passed"
        
        rule_name = "custom_rule"
        self.validator.register_rule(
            name=rule_name,
            check_func=custom_check,
            severity=Severity.INFO,
            description="Custom validation rule"
        )
        
        self.assertIn(rule_name, self.validator.rules)
        rule = self.validator.rules[rule_name]
        self.assertEqual(rule.name, rule_name)
        self.assertEqual(rule.severity, Severity.INFO)
        self.assertEqual(rule.description, "Custom validation rule")
    
    def test_validate_single_amendment(self):
        """Test validating a single amendment."""
        amendment = Mock()
        amendment.amendment_id = "amd_123"
        amendment.section_number = "45A"
        amendment.operation = "insert"
        amendment.text_content = "Insert new section 45A"
        
        report = self.validator.validate_single(amendment)
        
        self.assertIsInstance(report, ValidationReport)
        self.assertEqual(report.amendment_count, 1)
        self.assertGreater(len(report.sections), 0)
    
    def test_validate_multiple_amendments(self):
        """Test validating multiple amendments."""
        amendments = []
        for i in range(3):
            amendment = Mock()
            amendment.amendment_id = f"amd_{i}"
            amendment.section_number = f"45{i}"
            amendment.operation = "insert"
            amendments.append(amendment)
        
        report = self.validator.validate_multiple(amendments)
        
        self.assertIsInstance(report, ValidationReport)
        self.assertEqual(report.amendment_count, 3)
        self.assertGreater(len(report.sections), 0)
    
    def test_check_section_number_format_valid(self):
        """Test section number format check with valid input."""
        amendment = Mock()
        amendment.section_number = "45A"
        amendment.section_number_pattern = "45A"
        
        passed, message = self.validator._check_section_number_format(amendment, {})
        
        self.assertTrue(passed)
        self.assertIn("valid", message.lower())
    
    def test_check_section_number_format_invalid(self):
        """Test section number format check with invalid input."""
        amendment = Mock()
        amendment.section_number = "invalid-section"
        amendment.section_number_pattern = "invalid-section"
        
        passed, message = self.validator._check_section_number_format(amendment, {})
        
        self.assertFalse(passed)
        self.assertIn("invalid", message.lower())
    
    def test_check_operation_validity_valid(self):
        """Test operation validity check with valid operation."""
        amendment = Mock()
        amendment.operation = "insert"
        
        passed, message = self.validator._check_operation_validity(amendment, {})
        
        self.assertTrue(passed)
        self.assertIn("valid", message.lower())
    
    def test_check_operation_validity_invalid(self):
        """Test operation validity check with invalid operation."""
        amendment = Mock()
        amendment.operation = "invalid_op"
        
        passed, message = self.validator._check_operation_validity(amendment, {})
        
        self.assertFalse(passed)
        self.assertIn("invalid", message.lower())


if __name__ == '__main__':
    unittest.main()