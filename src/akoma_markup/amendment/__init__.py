"""Amendment-related functionality for akoma-markup."""

from .conversion import (
    build_gazette_chain,
    convert_gazette_text,
    convert_gazette_page_with_vision,
    convert_gazette_pages_with_vision,
    merge_gazette_pages,
)

__all__ = [
    "build_gazette_chain",
    "convert_gazette_text",
    "convert_gazette_page_with_vision",
    "convert_gazette_pages_with_vision",
    "merge_gazette_pages",
]
