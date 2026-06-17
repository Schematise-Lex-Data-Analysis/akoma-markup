"""Multimodal LLM factory for vision-based amendment extraction.

Provides clients for analyzing PDF page images to extract amendment annotations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from ...amendment.vision_schema import VisionExtractedAmendment
from ..pdf.image_renderer import PDFImageRenderer

logger = logging.getLogger(__name__)


@dataclass
class MultimodalLLMConfig:
    """Configuration for multimodal LLM clients."""
    
    provider: str  # "azure", "anthropic", "openai", etc.
    model: str  # Required - no defaults
    
    # Provider-specific fields
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    credential: Optional[str] = None
    
    # Common parameters
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # Image rendering parameters
    dpi: int = 120  # For PDF rendering
    image_detail: str = "high"  # "low", "high", or "auto" for vision models


class MultimodalLLMClient(ABC):
    """Abstract base class for multimodal LLM clients."""
    
    def __init__(self, config: MultimodalLLMConfig):
        self.config = config
        self.image_renderer = PDFImageRenderer(dpi=config.dpi)
    
    @abstractmethod
    async def analyze_page(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Analyze a page image with a custom prompt.
        
        Args:
            base64_image: Base64-encoded PNG image.
            prompt: Text prompt for analysis.
            
        Returns:
            Raw LLM response as dictionary.
        """
        pass
    
    @abstractmethod
    async def extract_amendments_from_page(self, base64_image: str, page_num: int) -> List[VisionExtractedAmendment]:
        """Extract amendments from a single page image.
        
        Args:
            base64_image: Base64-encoded PNG image.
            page_num: Page number for annotation.
            
        Returns:
            List of extracted amendments.
        """
        pass
    
    @abstractmethod
    async def classify_amendment_type(self, text_region: str, visual_context: str) -> Dict[str, Any]:
        """Classify amendment type from text and visual context.
        
        Args:
            text_region: Extracted text region.
            visual_context: Description of visual context.
            
        Returns:
            Classification result with type, confidence, etc.
        """
        pass


class AzureMultimodalClient(MultimodalLLMClient):
    """Azure AI Vision-based multimodal client."""
    
    def __init__(self, config: MultimodalLLMConfig):
        super().__init__(config)
        
        # Validate Azure-specific configuration
        if not config.endpoint:
            config.endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
        
        if not config.api_key:
            config.api_key = os.environ.get("AZURE_VISION_KEY")
        
        if not config.model:
            config.model = os.environ.get("AZURE_VISION_MODEL")
        
        if not config.endpoint or not config.api_key or not config.model:
            raise ValueError(
                "AzureMultimodalClient requires endpoint, api_key, and model "
                "(or AZURE_VISION_ENDPOINT / AZURE_VISION_KEY / "
                "AZURE_VISION_MODEL env vars)."
            )
        
        # Initialize Azure client
        self._client = self._create_azure_client()
    
    def _create_azure_client(self):
        """Create appropriate Azure client based on API style."""
        from .azure_api import validate_api_mode
        
        api_style = os.environ.get("AZURE_VISION_API_STYLE", "chat")
        api_mode = validate_api_mode(api_style, "AZURE_VISION_API_STYLE")
        
        if api_mode == "azure-inference":
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential
            
            return ChatCompletionsClient(
                endpoint=self.config.endpoint,
                credential=AzureKeyCredential(self.config.api_key),
            )
        else:
            # Use OpenAI-compatible client for chat/responses modes
            from openai import OpenAI
            
            return OpenAI(
                base_url=self.config.endpoint,
                api_key=self.config.api_key,
            )
    
    async def analyze_page(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Analyze page image with Azure vision model."""
        try:
            # Prepare the request
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": self.config.image_detail
                            }
                        }
                    ]
                }
            ]
            
            # Make the request
            if hasattr(self._client, 'chat'):
                # OpenAI-style client
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                return {
                    "text": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.dict() if response.usage else None,
                    "model": response.model
                }
            else:
                # Azure Inference client
                response = await asyncio.to_thread(
                    self._client.complete,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                return {
                    "text": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": self.config.model
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze page with Azure: {e}")
            raise
    
    async def extract_amendments_from_page(self, base64_image: str, page_num: int) -> List[VisionExtractedAmendment]:
        """Extract amendments from page using structured prompting."""
        
        # Create prompt for amendment extraction
        prompt = """Analyze this legislative document page and identify all amendments.
        
        Look for:
        1. Substitution amendments (words/text being replaced)
        2. Insertion amendments (new text being added)
        3. Omission amendments (text being deleted)
        4. Repeal amendments (sections being repealed)
        
        For each amendment found, provide:
        - Amendment type
        - Act number and year
        - Section number being amended
        - Target location (subsection, clause, etc.)
        - Original text being modified
        - New text (if substitution)
        - Effective date
        - Any footnote markers
        
        Format the response as a JSON array of amendment objects."""
        
        try:
            response = await self.analyze_page(base64_image, prompt)
            
            # Parse the JSON response
            amendments_data = self._parse_amendments_response(response["text"], page_num)
            
            # Convert to VisionExtractedAmendment objects
            amendments = []
            for am_data in amendments_data:
                try:
                    amendment = VisionExtractedAmendment(
                        page_num=page_num,
                        bbox=am_data.get("bbox", (0, 0, 0, 0)),  # Default bbox
                        amendment_type=am_data.get("amendment_type", "unknown"),
                        act_number=am_data.get("act_number", ""),
                        act_year=am_data.get("act_year", ""),
                        section_number=am_data.get("section_number", ""),
                        target_section=am_data.get("target_section", ""),
                        target_location=am_data.get("target_location", ""),
                        original_text=am_data.get("original_text", ""),
                        new_text=am_data.get("new_text"),
                        effective_date=am_data.get("effective_date", ""),
                        footnote_marker=am_data.get("footnote_marker", ""),
                        confidence_score=am_data.get("confidence_score", 1.0),
                        visual_context=am_data.get("visual_context", ""),
                        raw_llm_response=response
                    )
                    
                    if amendment.is_valid():
                        amendments.append(amendment)
                    else:
                        logger.warning(f"Invalid amendment detected on page {page_num}: {amendment}")
                        
                except Exception as e:
                    logger.error(f"Failed to parse amendment data: {e}")
            
            return amendments
            
        except Exception as e:
            logger.error(f"Failed to extract amendments from page {page_num}: {e}")
            return []
    
    async def classify_amendment_type(self, text_region: str, visual_context: str) -> Dict[str, Any]:
        """Classify amendment type using both text and visual context."""
        
        prompt = f"""Classify the amendment type based on the following text and visual context:

Text Region:
{text_region}

Visual Context:
{visual_context}

Classification options:
1. substitution - text is being replaced
2. insertion - new text is being added  
3. omission - text is being deleted
4. repeal - entire section is being repealed
5. addition - completely new section is being added

Respond with a JSON object containing:
- amendment_type: one of the above options
- confidence: float between 0 and 1
- reasoning: brief explanation
- keywords: list of keywords supporting classification
"""
        
        try:
            response = await self.analyze_page("", prompt)  # No image, just text
            
            # Parse classification
            try:
                classification = json.loads(response["text"])
                return classification
            except json.JSONDecodeError:
                # Fallback to simple parsing
                return {
                    "amendment_type": "unknown",
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response",
                    "keywords": []
                }
                
        except Exception as e:
            logger.error(f"Failed to classify amendment type: {e}")
            return {
                "amendment_type": "unknown",
                "confidence": 0.0,
                "reasoning": f"Classification failed: {str(e)}",
                "keywords": []
            }
    
    def _parse_amendments_response(self, response_text: str, page_num: int) -> List[Dict[str, Any]]:
        """Parse LLM response into amendment data structures."""
        try:
            # Try to extract JSON from response
            import re
            
            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                amendments_json = json_match.group(0)
                return json.loads(amendments_json)
            
            # If no JSON found, try to parse as plain text
            amendments = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and "amendment" in line.lower():
                    # Simple heuristic parsing
                    amendments.append({
                        "amendment_type": "unknown",
                        "act_number": "",
                        "act_year": "",
                        "section_number": "",
                        "target_section": "",
                        "target_location": "",
                        "original_text": line,
                        "confidence_score": 0.5
                    })
            
            return amendments
            
        except Exception as e:
            logger.error(f"Failed to parse amendments response: {e}")
            return []


class AnthropicMultimodalClient(MultimodalLLMClient):
    """Anthropic Claude multimodal client (when available)."""
    
    def __init__(self, config: MultimodalLLMConfig):
        super().__init__(config)
        
        # Check for Anthropic support
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic multimodal support requires anthropic package. "
                "Install with: pip install anthropic"
            )
        
        if not config.api_key:
            config.api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not config.api_key:
            raise ValueError(
                "AnthropicMultimodalClient requires api_key or ANTHROPIC_API_KEY env var."
            )
        
        self._client = anthropic.Anthropic(api_key=config.api_key)
    
    async def analyze_page(self, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Analyze page with Anthropic Claude."""
        try:
            response = await asyncio.to_thread(
                self._client.messages.create,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            return {
                "text": response.content[0].text,
                "finish_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "model": response.model
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze page with Anthropic: {e}")
            raise
    
    async def extract_amendments_from_page(self, base64_image: str, page_num: int) -> List[VisionExtractedAmendment]:
        """Extract amendments using Anthropic Claude."""
        # Similar implementation to Azure, using Anthropic's API
        raise NotImplementedError("Anthropic amendment extraction not yet implemented")
    
    async def classify_amendment_type(self, text_region: str, visual_context: str) -> Dict[str, Any]:
        """Classify amendment type using Anthropic."""
        raise NotImplementedError("Anthropic classification not yet implemented")


class MultimodalLLMFactory:
    """Factory for creating multimodal LLM clients."""
    
    @staticmethod
    def create_client(config: Union[Dict[str, Any], MultimodalLLMConfig]) -> MultimodalLLMClient:
        """Create a multimodal LLM client from configuration.
        
        Args:
            config: Either a dictionary or MultimodalLLMConfig object.
                   Must include 'provider' and 'model' fields.
        
        Returns:
            Configured MultimodalLLMClient instance.
        
        Raises:
            ValueError: If provider is not supported or config is invalid.
        """
        if isinstance(config, dict):
            config_obj = MultimodalLLMConfig(**config)
        else:
            config_obj = config
        
        if not config_obj.provider:
            raise ValueError("Multimodal LLM config must include a 'provider' field")
        
        if not config_obj.model:
            raise ValueError("Multimodal LLM config must include a 'model' field")
        
        provider = config_obj.provider.lower()
        
        if provider == "azure":
            return AzureMultimodalClient(config_obj)
        elif provider == "anthropic":
            return AnthropicMultimodalClient(config_obj)
        else:
            raise ValueError(f"Unsupported multimodal provider: {provider}. "
                           f"Supported: 'azure', 'anthropic'")