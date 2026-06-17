"""PDF Image Renderer for Multimodal Amendment Extraction.

Provides functionality to render PDF pages to base64-encoded images for vision LLM analysis.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from .images import render_page, render_pages


@dataclass
class PageDimensions:
    """Dimensions of a PDF page in points and pixels."""
    width_points: float  # Points (1 point = 1/72 inch)
    height_points: float
    width_pixels: int  # Rendered pixels at default DPI
    height_pixels: int
    dpi: int = 120  # DPI used for rendering


class PDFImageRenderer:
    """Renderer for converting PDF pages to base64-encoded images for vision LLMs."""
    
    def __init__(self, dpi: int = 120):
        """Initialize the renderer with a target DPI.
        
        Args:
            dpi: Resolution for rendering images. Higher DPI provides more detail
                 but increases image size. 120 DPI is suitable for vision LLMs.
        """
        self.dpi = dpi
    
    def render_page_to_base64(self, pdf_path: Path, page_num: int) -> str:
        """Render a PDF page to a base64-encoded PNG image.
        
        Args:
            pdf_path: Path to the PDF file.
            page_num: 1-indexed page number.
            
        Returns:
            Base64-encoded PNG image string (without data URL prefix).
        """
        img = render_page(pdf_path, page_num, dpi=self.dpi)
        return self._pil_to_base64(img)
    
    def render_page_region(self, pdf_path: Path, page_num: int, bbox: Tuple[float, float, float, float]) -> str:
        """Render a specific region of a PDF page to base64.
        
        Args:
            pdf_path: Path to the PDF file.
            page_num: 1-indexed page number.
            bbox: Region to extract in points (x0, y0, x1, y1) where (0,0) is bottom-left.
            
        Returns:
            Base64-encoded PNG image of the cropped region.
            
        Note:
            PDF coordinates use bottom-left as origin. The bbox should be specified
            in points (1 point = 1/72 inch).
        """
        # First render the full page
        img = render_page(pdf_path, page_num, dpi=self.dpi)
        
        # Get page dimensions to convert points to pixels
        dims = self.get_page_dimensions(pdf_path, page_num)
        
        # Convert points to pixels
        scale = self.dpi / 72.0
        x0_px = int(bbox[0] * scale)
        y0_px = int(bbox[1] * scale)
        x1_px = int(bbox[2] * scale)
        y1_px = int(bbox[3] * scale)
        
        # Flip y-coordinate (PDF origin is bottom-left, PIL is top-left)
        height_px = dims.height_pixels
        y0_px_flipped = height_px - y1_px
        y1_px_flipped = height_px - y0_px
        
        # Ensure coordinates are within bounds
        x0_px = max(0, min(x0_px, img.width - 1))
        x1_px = max(x0_px + 1, min(x1_px, img.width))
        y0_px_flipped = max(0, min(y0_px_flipped, img.height - 1))
        y1_px_flipped = max(y0_px_flipped + 1, min(y1_px_flipped, img.height))
        
        # Crop the image
        cropped_img = img.crop((x0_px, y0_px_flipped, x1_px, y1_px_flipped))
        
        return self._pil_to_base64(cropped_img)
    
    def get_page_dimensions(self, pdf_path: Path, page_num: int) -> PageDimensions:
        """Get the dimensions of a PDF page in points and rendered pixels.
        
        Args:
            pdf_path: Path to the PDF file.
            page_num: 1-indexed page number.
            
        Returns:
            PageDimensions object with width/height in points and pixels.
        """
        import pypdfium2 as pdfium
        
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if page_num < 1 or page_num > len(pdf):
                raise ValueError(
                    f"page_num {page_num} out of range (PDF has {len(pdf)} pages)"
                )
            
            page = pdf[page_num - 1]
            width_pt = page.get_width()
            height_pt = page.get_height()
            
            scale = self.dpi / 72.0
            width_px = int(width_pt * scale)
            height_px = int(height_pt * scale)
            
            return PageDimensions(
                width_points=width_pt,
                height_points=height_pt,
                width_pixels=width_px,
                height_pixels=height_px,
                dpi=self.dpi
            )
        finally:
            pdf.close()
    
    def _pil_to_base64(self, img: Image.Image, format: str = "PNG") -> str:
        """Convert a PIL image to base64-encoded string.
        
        Args:
            img: PIL Image object.
            format: Image format (PNG, JPEG, etc.).
            
        Returns:
            Base64-encoded image string without data URL prefix.
        """
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    
    def render_multiple_pages(self, pdf_path: Path, page_nums: list[int]) -> dict[int, str]:
        """Render multiple pages to base64 images in a single PDF open operation.
        
        Args:
            pdf_path: Path to the PDF file.
            page_nums: List of 1-indexed page numbers.
            
        Returns:
            Dictionary mapping page numbers to base64-encoded images.
        """
        images = render_pages(pdf_path, page_nums, dpi=self.dpi)
        return {page_num: self._pil_to_base64(img) for page_num, img in images.items()}