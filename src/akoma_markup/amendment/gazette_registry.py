"""Gazette registry for linking amendments to gazette notifications.

This module provides functionality to create and query a registry of
gazette notifications, enabling linkage between amendments and their
official gazette references.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class GazetteEntry:
    """Represents a gazette notification entry."""
    
    gazette_number: str
    gazette_date: str
    act_name: str
    act_year: str
    file_path: Path
    page_numbers: str = ""
    additional_info: str = ""

    @property
    def act_reference(self) -> str:
        """Get act reference in format 'Act X of YYYY'."""
        # Extract act number from act_name if possible
        match = re.search(r'Act\s+(\d+)\s+of\s+(\d{4})', self.act_name, re.IGNORECASE)
        if match:
            return f"Act {match.group(1)} of {match.group(2)}"
        
        # Try to extract from filename or additional info
        match = re.search(r'(\d+)\s+of\s+(\d{4})', self.act_name, re.IGNORECASE)
        if match:
            return f"Act {match.group(1)} of {match.group(2)}"
        
        return self.act_name


class GazetteRegistry:
    """Registry for gazette notifications."""
    
    def __init__(self):
        self.entries: List[GazetteEntry] = []
        self._act_to_gazette: Dict[str, List[GazetteEntry]] = {}
    
    def add_entry(self, entry: GazetteEntry) -> None:
        """Add a gazette entry to the registry."""
        self.entries.append(entry)
        act_ref = entry.act_reference
        if act_ref not in self._act_to_gazette:
            self._act_to_gazette[act_ref] = []
        self._act_to_gazette[act_ref].append(entry)
    
    def find_gazette_for_act(self, act_reference: str) -> Optional[GazetteEntry]:
        """Find gazette entry for a given act reference.
        
        Args:
            act_reference: Act reference in format 'Act X of YYYY' or similar
            
        Returns:
            GazetteEntry if found, None otherwise
        """
        # Direct match
        if act_reference in self._act_to_gazette:
            entries = self._act_to_gazette[act_reference]
            if entries:
                return entries[0]  # Return first match
        
        # Try to normalize act reference
        normalized = self._normalize_act_reference(act_reference)
        if normalized in self._act_to_gazette:
            entries = self._act_to_gazette[normalized]
            if entries:
                return entries[0]
        
        return None
    
    def _normalize_act_reference(self, act_ref: str) -> str:
        """Normalize act reference to standard format."""
        # Remove extra spaces and standardize
        act_ref = re.sub(r'\s+', ' ', act_ref.strip())
        
        # Try to extract Act X of YYYY pattern
        match = re.search(r'Act\s+(\d+)\s+of\s+(\d{4})', act_ref, re.IGNORECASE)
        if match:
            return f"Act {match.group(1)} of {match.group(2)}"
        
        return act_ref
    
    def load_from_directory(self, directory: Path) -> None:
        """Load gazette entries from a directory of PDF files.
        
        This method attempts to extract gazette information from filenames.
        For production use, a proper gazette extraction pipeline should be used.
        """
        for pdf_file in directory.rglob("*.pdf"):  # rglob for recursive search
            self._add_from_filename(pdf_file)
    
    def _add_from_filename(self, pdf_path: Path) -> None:
        """Create GazetteEntry from filename."""
        filename = pdf_path.name
        
        # Try to extract gazette number (6 digits)
        gazette_match = re.search(r'(\d{6})', filename)
        gazette_number = gazette_match.group(1) if gazette_match else ""
        
        # Try to extract date (look for patterns like 2023, 2022, 28Oct2022)
        date_match = re.search(r'(?:(\d{1,2})(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(\d{4})|\b(\d{4})\b)', filename, re.IGNORECASE)
        gazette_date = ""
        if date_match:
            if date_match.group(3):  # Just year
                gazette_date = date_match.group(3)
            else:  # DDMonYYYY
                gazette_date = f"{date_match.group(1)}-{date_match.group(2)[:3]}-{date_match.group(2)}"
        
        # Try to extract act name
        act_name = "IT Act Rules Amendment"
        act_year = "2023" if "2023" in filename else "2022" if "2022" in filename else "2021"
        
        # Handle specific filename patterns
        if "IT-Rules-Amendment" in filename:
            act_name = "IT Act Rules Amendment"
        elif "IT-Amendment-Rules" in filename:
            act_name = "IT Amendment Rules"
        elif "Intermediary_Guidelines" in filename:
            act_name = "Intermediary Guidelines and Digital Media Ethics Code Rules, 2021"
        elif "IT_amendment_act2008" in filename:
            # This is likely the gazette for Act 10 of 2009
            act_name = "Act 10 of 2009"
            act_year = "2009"
        elif "it_amendment_act2008" in filename.lower():
            # Case insensitive match
            act_name = "Act 10 of 2009"
            act_year = "2009"
        
        entry = GazetteEntry(
            gazette_number=gazette_number,
            gazette_date=gazette_date,
            act_name=act_name,
            act_year=act_year,
            file_path=pdf_path,
            additional_info=f"Extracted from filename: {filename}"
        )
        
        self.add_entry(entry)
    
    def save_to_csv(self, output_path: Path) -> None:
        """Save registry to CSV file."""
        fieldnames = [
            "gazette_number",
            "gazette_date",
            "act_name",
            "act_year",
            "file_path",
            "page_numbers",
            "additional_info",
            "act_reference"
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for entry in self.entries:
                writer.writerow({
                    "gazette_number": entry.gazette_number,
                    "gazette_date": entry.gazette_date,
                    "act_name": entry.act_name,
                    "act_year": entry.act_year,
                    "file_path": str(entry.file_path),
                    "page_numbers": entry.page_numbers,
                    "additional_info": entry.additional_info,
                    "act_reference": entry.act_reference
                })


def create_gazette_registry(gazette_dir: Path) -> GazetteRegistry:
    """Create a gazette registry from a directory of gazette PDFs.
    
    Args:
        gazette_dir: Directory containing gazette notification PDFs
        
    Returns:
        GazetteRegistry populated with entries
    """
    registry = GazetteRegistry()
    if gazette_dir.exists():
        registry.load_from_directory(gazette_dir)
    return registry


def link_amendments_to_gazettes(
    amendments: List,
    gazette_dir: Path,
    output_tsv: Optional[Path] = None
) -> Dict[str, str]:
    """Link amendments to gazette references.
    
    Args:
        amendments: List of amendment objects with amendment_act_id attribute
        gazette_dir: Directory containing gazette notifications
        output_tsv: Optional path to output updated TSV
        
    Returns:
        Dictionary mapping amendment_act_id to gazette reference
    """
    registry = create_gazette_registry(gazette_dir)
    
    # Create mapping
    mapping = {}
    for amdt in amendments:
        act_id = getattr(amdt, 'amendment_act_id', None)
        if not act_id:
            continue
        
        entry = registry.find_gazette_for_act(act_id)
        if entry:
            # Format gazette reference
            ref_parts = []
            if entry.gazette_number:
                ref_parts.append(f"Gazette No. {entry.gazette_number}")
            if entry.gazette_date:
                ref_parts.append(f"dated {entry.gazette_date}")
            
            gazette_ref = ", ".join(ref_parts) if ref_parts else "Gazette notification"
            mapping[act_id] = gazette_ref
        else:
            mapping[act_id] = ""
    
    return mapping