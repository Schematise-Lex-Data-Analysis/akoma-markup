"""Content processing chain for Gazette verification."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Any


class ContentProcessor(ABC):
    """Base class for content processors."""
    
    @abstractmethod
    def process(self, content: str) -> str:
        """Process content and return cleaned version."""
        pass
    
    def __call__(self, content: str) -> str:
        """Make processors callable."""
        return self.process(content)


class BracketCleanupProcessor(ContentProcessor):
    """Remove unnecessary square brackets from Gazette content."""
    
    def __init__(self, preserve_table_placeholders: bool = True):
        """Initialize bracket cleanup processor.
        
        Args:
            preserve_table_placeholders: Whether to preserve ``<<TABLE_REGION:N>>`` markers
        """
        self.preserve_table_placeholders = preserve_table_placeholders
        
        # Patterns to remove (structural markers)
        self.remove_patterns = [
            # Section markers: [1.], [2.], [3.], etc.
            r'^\[\d+\.\](?:\s|$)',
            r'(?<=\n)\[\d+\.\](?:\s|$)',
            
            # Subsection markers: [(1)], [(2)], [(a)], [(i)]
            r'^\[\(\d+\)\](?:\s|$)',
            r'^\[\([a-z]\)\](?:\s|$)',
            r'^\[\([ivxlcdm]+\)\](?:\s|$)',
            r'(?<=\n)\[\(\d+\)\](?:\s|$)',
            r'(?<=\n)\[\([a-z]\)\](?:\s|$)',
            r'(?<=\n)\[\([ivxlcdm]+\)\](?:\s|$)',
            
            # Chapter markers: [CHAPTER I], [CHAPTER II]
            r'^\[CHAPTER [IVXLCDM]+\](?:\s|$)',
            r'(?<=\n)\[CHAPTER [IVXLCDM]+\](?:\s|$)',
            
            # Standalone brackets that are likely artifacts
            r'^\[\](?:\s|$)',
            r'(?<=\n)\[\](?:\s|$)',
            
            # Duplicate markers (same marker multiple times)
            r'(\[[^]]+\])(?:\s*\1)+',
        ]
        
        # Patterns to preserve
        self.preserve_patterns = [
            # Table placeholders
            r'<<TABLE_REGION:\d+>>',
            
            # Citations: [Section 3], [Article 5]
            r'\[(?:Section|Article|Rule|Regulation|Schedule|Chapter|Part)\s+\S+\]',
            
            # Explanatory text in brackets
            r'\[[^]]{10,}\]',  # Brackets with significant content
        ]
    
    def process(self, content: str) -> str:
        """Remove unnecessary brackets while preserving important ones."""
        if not content:
            return content
        
        # Split into lines for line-by-line processing
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            processed_line = line
            
            # Check if line contains patterns to preserve
            should_preserve = False
            for pattern in self.preserve_patterns:
                if re.search(pattern, processed_line, re.IGNORECASE):
                    should_preserve = True
                    break
            
            if not should_preserve:
                # Apply removal patterns
                for pattern in self.remove_patterns:
                    processed_line = re.sub(pattern, '', processed_line)
                
                # Also handle inline brackets (not at start of line)
                # Remove standalone bracket markers within text
                processed_line = re.sub(r'\s+\[\d+\.\]\s+', ' ', processed_line)
                processed_line = re.sub(r'\s+\[\(\d+\)\]\s+', ' ', processed_line)
                processed_line = re.sub(r'\s+\[CHAPTER [IVXLCDM]+\]\s+', ' ', processed_line)
            
            # Clean up extra spaces
            processed_line = re.sub(r'\s+', ' ', processed_line).strip()
            
            if processed_line:  # Only add non-empty lines
                processed_lines.append(processed_line)
        
        return '\n'.join(processed_lines)


class WhitespaceNormalizer(ContentProcessor):
    """Normalize whitespace and line breaks."""
    
    def process(self, content: str) -> str:
        """Normalize whitespace in content."""
        if not content:
            return content
        
        # Replace multiple spaces with single space
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in content.split('\n')]
        
        # Remove consecutive blank lines (keep at most 2)
        processed_lines = []
        blank_line_count = 0
        for line in lines:
            if not line.strip():
                blank_line_count += 1
                if blank_line_count <= 2:
                    processed_lines.append(line)
            else:
                blank_line_count = 0
                processed_lines.append(line)
        
        # Ensure proper spacing around section markers
        content = '\n'.join(processed_lines)
        
        # Fix spacing after section numbers
        content = re.sub(r'(\d+\.)(\S)', r'\1 \2', content)  # 1.Text -> 1. Text
        content = re.sub(r'(\(\d+\))(\S)', r'\1 \2', content)  # (1)Text -> (1) Text
        content = re.sub(r'(\([a-z]\))(\S)', r'\1 \2', content)  # (a)Text -> (a) Text
        
        return content


class SectionMarkerValidator(ContentProcessor):
    """Validate and normalize section numbering."""
    
    def process(self, content: str) -> str:
        """Validate section numbering hierarchy."""
        if not content:
            return content
        
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            processed_line = line
            
            # Normalize section number formats
            # Convert "1" to "1."
            processed_line = re.sub(r'^(\d+)(?![\.\)])', r'\1.', processed_line)
            # Convert "1)" to "(1)"
            processed_line = re.sub(r'^(\d+)\)', r'(\1)', processed_line)
            
            processed_lines.append(processed_line)
        
        return '\n'.join(processed_lines)


class ContentProcessorChain:
    """Chain of content processors."""
    
    def __init__(self, processors: Optional[List[ContentProcessor]] = None):
        """Initialize processor chain.
        
        Args:
            processors: List of processors to apply in order
        """
        if processors is None:
            processors = [
                BracketCleanupProcessor(),
                WhitespaceNormalizer(),
                SectionMarkerValidator(),
            ]
        self.processors = processors
    
    def process(self, content: str) -> str:
        """Apply all processors in sequence."""
        for processor in self.processors:
            content = processor.process(content)
        return content
    
    def add_processor(self, processor: ContentProcessor, index: Optional[int] = None):
        """Add a processor to the chain.
        
        Args:
            processor: Processor to add
            index: Position to insert (None for end)
        """
        if index is None:
            self.processors.append(processor)
        else:
            self.processors.insert(index, processor)
    
    def remove_processor(self, processor_type: type) -> bool:
        """Remove first processor of given type.
        
        Args:
            processor_type: Type of processor to remove
            
        Returns:
            True if removed, False if not found
        """
        for i, processor in enumerate(self.processors):
            if isinstance(processor, processor_type):
                self.processors.pop(i)
                return True
        return False