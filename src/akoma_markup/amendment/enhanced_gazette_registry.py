"""Enhanced gazette registry with visual analysis integration.

Extends the basic GazetteRegistry with visual analysis capabilities
for improved gazette metadata extraction and amendment matching.
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import asyncio

from .gazette_registry import GazetteRegistry, GazetteEntry
from .visual_gazette import VisualGazetteAnalyzer, GazetteAnalysis
from .gazette_matcher import GazetteAmendmentMatcher, MatchingResults

logger = logging.getLogger(__name__)


@dataclass
class EnhancedGazetteEntry(GazetteEntry):
    """Enhanced gazette entry with visual analysis data."""
    
    visual_analysis: Optional[Dict[str, Any]] = None
    analysis_score: float = 0.0
    extraction_method: str = "filename"  # filename, visual, hybrid
    
    @classmethod
    def from_analysis(cls, pdf_path: Path, analysis: GazetteAnalysis) -> "EnhancedGazetteEntry":
        """Create enhanced entry from visual analysis.
        
        Args:
            pdf_path: Path to gazette PDF.
            analysis: GazetteAnalysis object.
            
        Returns:
            EnhancedGazetteEntry.
        """
        metadata = analysis.metadata
        
        entry = cls(
            gazette_number=metadata.get("gazette_number", ""),
            gazette_date=metadata.get("publication_date", ""),
            act_name=metadata.get("act_name", ""),
            act_year=metadata.get("act_year", ""),
            file_path=pdf_path,
            page_numbers="",  # Could be extracted from analysis
            additional_info=f"Visual analysis: {metadata.get('analysis_method', 'unknown')}",
            visual_analysis=analysis.to_dict(),
            analysis_score=analysis.validation.get("score", 0.0) if analysis.validation else 0.0,
            extraction_method="visual"
        )
        
        return entry


class EnhancedGazetteRegistry(GazetteRegistry):
    """Extended gazette registry with visual analysis capabilities."""
    
    def __init__(self, enable_visual_analysis: bool = True):
        """Initialize enhanced gazette registry.
        
        Args:
            enable_visual_analysis: Whether to enable visual analysis.
        """
        super().__init__()
        self.enable_visual_analysis = enable_visual_analysis
        self.visual_analyzer: Optional[VisualGazetteAnalyzer] = None
        self.matcher: Optional[GazetteAmendmentMatcher] = None
        
        if enable_visual_analysis:
            self.visual_analyzer = VisualGazetteAnalyzer(enable_ocr=True, cache_results=True)
            self.matcher = GazetteAmendmentMatcher()
        
        self.analysis_cache: Dict[str, GazetteAnalysis] = {}
    
    async def load_from_directory(
        self, 
        directory: Path, 
        use_vision: bool = True,
        analysis_cache_file: Optional[Path] = None
    ) -> None:
        """Load gazette entries from directory with optional visual analysis.
        
        Args:
            directory: Directory containing gazette PDFs.
            use_vision: Whether to use visual analysis.
            analysis_cache_file: Optional file to load/save analysis cache.
        """
        logger.info(f"Loading gazettes from {directory} (use_vision: {use_vision})")
        
        # Load analysis cache if available
        if analysis_cache_file and analysis_cache_file.exists():
            try:
                self._load_analysis_cache(analysis_cache_file)
                logger.info(f"Loaded analysis cache from {analysis_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load analysis cache: {e}")
        
        # Find PDF files
        pdf_files = list(directory.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process each file
        for pdf_file in pdf_files:
            try:
                if use_vision and self.visual_analyzer:
                    # Use visual analysis
                    await self._add_from_visual_analysis(pdf_file)
                else:
                    # Use filename-based extraction (original method)
                    self._add_from_filename(pdf_file)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file}: {e}")
                # Fallback to filename-based extraction
                try:
                    self._add_from_filename(pdf_file)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {pdf_file}: {fallback_error}")
        
        # Save analysis cache if requested
        if analysis_cache_file and self.analysis_cache:
            try:
                self._save_analysis_cache(analysis_cache_file)
                logger.info(f"Saved analysis cache to {analysis_cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save analysis cache: {e}")
        
        logger.info(f"Registry loaded with {len(self.entries)} entries")
    
    async def _add_from_visual_analysis(self, pdf_path: Path) -> None:
        """Add gazette entry using visual analysis.
        
        Args:
            pdf_path: Path to gazette PDF.
        """
        if not self.visual_analyzer:
            raise ValueError("Visual analyzer not initialized")
        
        # Check cache first
        cache_key = str(pdf_path)
        if cache_key in self.analysis_cache:
            analysis = self.analysis_cache[cache_key]
        else:
            # Perform analysis
            analysis = await self.visual_analyzer.analyze_gazette(pdf_path)
            self.analysis_cache[cache_key] = analysis
        
        # Create enhanced entry
        entry = EnhancedGazetteEntry.from_analysis(pdf_path, analysis)
        self.add_entry(entry)
        
        logger.debug(f"Added gazette from visual analysis: {pdf_path.name}")
    
    def _load_analysis_cache(self, cache_file: Path) -> None:
        """Load analysis cache from file.
        
        Args:
            cache_file: Path to cache file.
        """
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        for path_str, analysis_data in cache_data.items():
            analysis = GazetteAnalysis.from_dict(analysis_data)
            self.analysis_cache[path_str] = analysis
    
    def _save_analysis_cache(self, cache_file: Path) -> None:
        """Save analysis cache to file.
        
        Args:
            cache_file: Path to cache file.
        """
        cache_data = {}
        for path_str, analysis in self.analysis_cache.items():
            cache_data[path_str] = analysis.to_dict()
        
        # Ensure parent directory exists
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    def find_gazette_for_amendment(self, amendment: Any) -> Optional[EnhancedGazetteEntry]:
        """Find best matching gazette for an amendment.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            EnhancedGazetteEntry if found, None otherwise.
        """
        if not self.matcher:
            # Fallback to basic matching
            return self._find_gazette_basic(amendment)
        
        # Get all gazette analyses
        gazette_analyses = []
        for entry in self.entries:
            if isinstance(entry, EnhancedGazetteEntry) and entry.visual_analysis:
                analysis = GazetteAnalysis.from_dict(entry.visual_analysis)
                gazette_analyses.append(analysis)
            else:
                # Create minimal analysis for basic entries
                analysis = GazetteAnalysis(
                    entry.file_path,
                    metadata={
                        "act_name": entry.act_name,
                        "act_year": entry.act_year,
                        "act_number": self._extract_act_number(entry.act_name),
                        "gazette_number": entry.gazette_number,
                        "publication_date": entry.gazette_date,
                        "filename": entry.file_path.name,
                        "extraction_method": "filename"
                    }
                )
                gazette_analyses.append(analysis)
        
        # Match amendment to gazettes
        matches = self.matcher.match_amendments_to_gazettes([amendment], gazette_analyses)
        
        if matches.matches:
            # Get the best match
            best_match = matches.matches[0]
            
            # Find corresponding entry
            for entry in self.entries:
                if str(entry.file_path) == str(best_match.gazette.pdf_path):
                    return entry
        
        return None
    
    def _find_gazette_basic(self, amendment: Any) -> Optional[EnhancedGazetteEntry]:
        """Basic gazette finding without visual analysis.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            EnhancedGazetteEntry if found, None otherwise.
        """
        # Extract amendment information
        amendment_data = self._extract_amendment_data(amendment)
        
        if not amendment_data:
            return None
        
        # First try exact act/year match
        exact_matches = []
        for entry in self.entries:
            if (entry.act_year == amendment_data.get("act_year") and 
                self._act_numbers_match(entry.act_name, amendment_data.get("act_number"))):
                exact_matches.append(entry)
        
        if exact_matches:
            return self._select_best_match(amendment_data, exact_matches)
        
        # Try fuzzy match by act/year
        fuzzy_matches = []
        for entry in self.entries:
            match_score = self._calculate_basic_match_score(amendment_data, entry)
            if match_score >= 0.7:  # 70% match threshold
                fuzzy_matches.append((entry, match_score))
        
        if fuzzy_matches:
            # Return highest scoring match
            fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
            return fuzzy_matches[0][0]
        
        # No match found
        return None
    
    def _extract_amendment_data(self, amendment: Any) -> Dict[str, Any]:
        """Extract data from amendment object.
        
        Args:
            amendment: Amendment object.
            
        Returns:
            Dictionary with amendment data.
        """
        data = {}
        
        # Try various attribute patterns
        attr_patterns = [
            ("act_number", ["act_number", "amendment_act_id", "target_act_number"]),
            ("act_year", ["act_year", "amendment_year", "target_year"]),
            ("section_number", ["section_number", "target_section", "section"]),
            ("effective_date", ["effective_date", "date", "notification_date"]),
        ]
        
        for target_key, source_attrs in attr_patterns:
            for source_attr in source_attrs:
                if hasattr(amendment, source_attr):
                    value = getattr(amendment, source_attr)
                    if value:
                        data[target_key] = str(value)
                        break
        
        return data
    
    def _act_numbers_match(self, entry_act_name: str, amendment_act_number: Optional[str]) -> bool:
        """Check if act numbers match.
        
        Args:
            entry_act_name: Entry act name.
            amendment_act_number: Amendment act number.
            
        Returns:
            True if act numbers match.
        """
        if not amendment_act_number:
            return False
        
        # Extract act number from entry act name
        entry_act_number = self._extract_act_number(entry_act_name)
        
        return entry_act_number == amendment_act_number
    
    def _extract_act_number(self, act_ref: str) -> Optional[str]:
        """Extract act number from reference string.
        
        Args:
            act_ref: Act reference string.
            
        Returns:
            Extracted act number, or None.
        """
        import re
        
        if not act_ref:
            return None
        
        # Look for Act X pattern
        match = re.search(r'Act\s+(\d+)', act_ref, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for standalone number
        match = re.search(r'\b(\d{1,4})\b', act_ref)
        if match:
            return match.group(1)
        
        return None
    
    def _calculate_basic_match_score(
        self, 
        amendment_data: Dict[str, Any], 
        entry: GazetteEntry
    ) -> float:
        """Calculate basic match score.
        
        Args:
            amendment_data: Amendment data dictionary.
            entry: Gazette entry.
            
        Returns:
            Match score between 0 and 1.
        """
        score = 0.0
        
        # Year match (40% weight)
        if amendment_data.get("act_year") == entry.act_year:
            score += 0.4
        
        # Act number match (30% weight)
        if (amendment_data.get("act_number") and 
            self._act_numbers_match(entry.act_name, amendment_data["act_number"])):
            score += 0.3
        
        # Date proximity (20% weight)
        if amendment_data.get("effective_date") and entry.gazette_date:
            date_score = self._calculate_date_proximity(
                amendment_data["effective_date"], 
                entry.gazette_date
            )
            score += 0.2 * date_score
        
        # Gazette number similarity (10% weight)
        if amendment_data.get("gazette_number") and entry.gazette_number:
            if amendment_data["gazette_number"] == entry.gazette_number:
                score += 0.1
        
        return score
    
    def _calculate_date_proximity(self, date1: str, date2: str) -> float:
        """Calculate date proximity score.
        
        Args:
            date1: First date string.
            date2: Second date string.
            
        Returns:
            Proximity score between 0 and 1.
        """
        import re
        from datetime import datetime
        
        try:
            # Extract years
            year1 = re.search(r'\b(\d{4})\b', date1)
            year2 = re.search(r'\b(\d{4})\b', date2)
            
            if year1 and year2:
                y1 = int(year1.group(1))
                y2 = int(year2.group(1))
                
                if y1 == y2:
                    return 1.0
                elif abs(y1 - y2) == 1:
                    return 0.5
                else:
                    return 0.1
        except Exception:
            pass
        
        return 0.0
    
    def _select_best_match(
        self, 
        amendment_data: Dict[str, Any], 
        matches: List[EnhancedGazetteEntry]
    ) -> EnhancedGazetteEntry:
        """Select best match from multiple candidates.
        
        Args:
            amendment_data: Amendment data.
            matches: List of candidate entries.
            
        Returns:
            Best matching entry.
        """
        if len(matches) == 1:
            return matches[0]
        
        # Score each match
        scored_matches = []
        for entry in matches:
            score = self._calculate_basic_match_score(amendment_data, entry)
            
            # Boost score for enhanced entries with good analysis
            if isinstance(entry, EnhancedGazetteEntry):
                if entry.analysis_score > 0.8:
                    score *= 1.2
                elif entry.analysis_score > 0.6:
                    score *= 1.1
            
            scored_matches.append((entry, score))
        
        # Return highest scoring match
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        return scored_matches[0][0]
    
    def match_all_amendments(
        self, 
        amendments: List[Any],
        output_file: Optional[Path] = None
    ) -> MatchingResults:
        """Match all amendments to gazettes in registry.
        
        Args:
            amendments: List of amendment objects.
            output_file: Optional file to save matching results.
            
        Returns:
            MatchingResults with all matches.
        """
        if not self.matcher:
            raise ValueError("Matcher not initialized")
        
        logger.info(f"Matching {len(amendments)} amendments to {len(self.entries)} gazettes")
        
        # Convert entries to analyses
        gazette_analyses = []
        for entry in self.entries:
            if isinstance(entry, EnhancedGazetteEntry) and entry.visual_analysis:
                analysis = GazetteAnalysis.from_dict(entry.visual_analysis)
            else:
                # Create basic analysis
                analysis = GazetteAnalysis(
                    entry.file_path,
                    metadata={
                        "act_name": entry.act_name,
                        "act_year": entry.act_year,
                        "gazette_number": entry.gazette_number,
                        "publication_date": entry.gazette_date,
                        "filename": entry.file_path.name
                    }
                )
            gazette_analyses.append(analysis)
        
        # Perform matching
        results = self.matcher.match_amendments_to_gazettes(amendments, gazette_analyses)
        
        # Save results if requested
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved matching results to {output_file}")
        
        return results


def create_enhanced_gazette_registry(
    gazette_dir: Path,
    use_vision: bool = True,
    analysis_cache_file: Optional[Path] = None
) -> EnhancedGazetteRegistry:
    """Create enhanced gazette registry from directory.
    
    Args:
        gazette_dir: Directory containing gazette PDFs.
        use_vision: Whether to use visual analysis.
        analysis_cache_file: Optional cache file for analysis results.
        
    Returns:
        EnhancedGazetteRegistry populated with entries.
    """
    registry = EnhancedGazetteRegistry(enable_visual_analysis=use_vision)
    
    # Run async loading
    async def load_registry():
        await registry.load_from_directory(gazette_dir, use_vision, analysis_cache_file)
    
    asyncio.run(load_registry())
    
    return registry