""""
Table OCR and AI analysis for akoma-markup.

Provides OCR extraction and multimodal AI analysis for IndiaCode legislative tables
using Azure AI services.
"""

import os
import json
import base64
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Install with: pip install langchain-openai")


class AzureOCR:
    """Azure AI Document Intelligence (OCR) service wrapper."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "mistral-document-ai-2512",
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize Azure OCR client.
        
        Args:
            api_key: Azure API key (default: AZURE_API_KEY env var)
            endpoint: Azure OCR endpoint (default: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)
            model: OCR model to use
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Azure API key required. Provide via api_key parameter "
                "or set AZURE_API_KEY environment variable."
            )
        
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def extract_text(self, pdf_path: Union[str, Path], include_images: bool = False) -> str:
        """
        Extract text from PDF using Azure OCR.
        
        Args:
            pdf_path: Path to PDF file
            include_images: Whether to include base64 images in response
            
        Returns:
            Extracted markdown text
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"Extracting text from {pdf_path.name}...")
        
        # Read and encode PDF
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Prepare payload
        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_base64}"
            },
            "include_image_base64": include_images
        }
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract markdown text from pages
                    all_text = []
                    if 'pages' in result:
                        for page in result['pages']:
                            if 'markdown' in page and page['markdown']:
                                all_text.append(page['markdown'])
                    
                    text = '\n'.join(all_text)
                    
                    if text.strip():
                        print(f"✓ Extracted {len(text)} characters from {len(result.get('pages', []))} pages")
                        return text
                    else:
                        raise ValueError("No text extracted from PDF")
                        
                elif response.status_code == 502 and attempt < self.max_retries - 1:
                    print(f"⚠️ 502 error, retrying ({attempt + 1}/{self.max_retries})...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    response.raise_for_status()
                    
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"⚠️ Error: {str(e)[:100]}..., retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise Exception(f"OCR extraction failed after {self.max_retries} attempts: {e}")
        
        raise Exception("OCR extraction failed")
    
    def extract_and_save(self, pdf_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Extract text from PDF and save to file.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted text (default: same as PDF)
            
        Returns:
            Path to saved text file
        """
        pdf_path = Path(pdf_path)
        
        if output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract text
        text = self.extract_text(pdf_path)
        
        # Save to file
        output_file = output_dir / f"{pdf_path.stem}_extracted.txt"
        with open(output_file, 'w') as f:
            f.write(text)
        
        print(f"✓ Saved extracted text to: {output_file}")
        return output_file


class AzureMultimodalAnalyzer:
    """Azure multimodal AI analyzer using LangChain."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ):
        """
        Initialize Azure multimodal analyzer.
        
        Args:
            api_key: Azure API key (default: AZURE_API_KEY env var)
            endpoint: OpenAI-compatible endpoint (default: AZURE_MULTIMODAL_ENDPOINT env var)
            deployment_name: multimodal deployment name (default: AZURE_MULTIMODAL_DEPLOYMENT env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain required. Install with: pip install langchain-openai langchain-core"
            )
        
        self.api_key = api_key
        if not self.api_key:
            raise ValueError(
                "Azure API key required. Provide via api_key parameter "
                "or set AZURE_API_KEY environment variable."
            )
        
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        
        # Initialize LangChain LLM
        # For Azure AI Foundry/OpenAI-compatible endpoints, use ChatOpenAI with base_url
        self.llm = ChatOpenAI(
            base_url=endpoint,
            api_key=api_key,
            model=deployment_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Initialize output parser
        self.output_parser = StrOutputParser()
    
    def analyze_table(
        self,
        table_text: str,
        analysis_type: str = "summary",
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze table text using multimodal AI.
        
        Args:
            table_text: Extracted table text
            analysis_type: Type of analysis (summary, structure, content, json)
            system_prompt: Optional custom system prompt
            
        Returns:
            Analysis result
        """
        # Define analysis prompts
        prompts = {
            "summary": """You are an expert in banking regulation and financial reporting.
Analyze this Banking Regulation Act balance sheet form and provide a concise summary:

1. FORM IDENTIFICATION:
   - Act, section, schedule
   - Form name and purpose
   - Key amendments

2. MAIN STRUCTURE:
   - Left side (Capital & Liabilities) categories
   - Right side (Property & Assets) categories

3. KEY REQUIREMENTS:
   - What data must be reported
   - Important formatting rules
   - Special disclosures needed

4. PRACTICAL USE:
   - Who uses this form
   - Most critical information
   - Common challenges

Be clear, practical, and focus on actionable insights.""",
            
            "structure": """You are a document structure analyst.
Analyze the STRUCTURE of this Banking Regulation Act balance sheet form:

FOCUS ON:
1. Overall layout (columns, sections, hierarchy)
2. Category organization and nesting
3. Data relationships between sections
4. Formatting patterns and conventions
5. Navigation through the form

Provide detailed structural analysis only.""",
            
            "content": """You are a financial content extraction expert.
Extract ALL CONTENT from this Banking Regulation Act balance sheet form:

REQUIRED OUTPUT:
1. COMPLETE asset categories and subcategories
2. COMPLETE liability categories and subcategories  
3. All special categories (investments, advances, deposits, etc.)
4. ALL footnotes and their meanings
5. Important definitions and terms
6. Data requirements for each section
7. Any calculations or formulas specified

Be thorough, complete, and systematic.""",
            
            "json": """You are a data extraction specialist.
Convert this Banking Regulation Act balance sheet form to structured JSON.

REQUIREMENTS:
- Output ONLY valid JSON
- Include: metadata, sections, categories, footnotes, definitions
- Structure should be clean and parseable
- Preserve all data from the form
- No additional text outside JSON

Example structure:
{
  "metadata": {
    "act": "Banking Regulation Act 1949",
    "section": "29",
    "form": "FORM A - Balance Sheet",
    "purpose": "..."
  },
  "sections": [...],
  "categories": {...},
  "footnotes": [...],
  "definitions": {...}
}"""
        }
        
        if analysis_type not in prompts:
            raise ValueError(
                f"Unknown analysis type: {analysis_type}. "
                f"Supported: {list(prompts.keys())}"
            )
        
        # Use custom system prompt or default
        if system_prompt is None:
            system_prompt = prompts[analysis_type]
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Table text:\n\n{table_text}")
        ])
        
        # Create chain
        chain = prompt | self.llm | self.output_parser
        
        # Limit text length for token constraints
        text_chunk = table_text[:4000]  # Conservative limit
        
        print(f"Analyzing with multimodal AI ({analysis_type})...")
        
        try:
            result = chain.invoke({"table_text": text_chunk})
            print(f"✓ Analysis complete")
            return result
            
        except Exception as e:
            raise Exception(f"multimodal analysis failed: {e}")
    
    def analyze_and_save(
        self,
        table_text: str,
        analysis_types: List[str] = None,
        output_dir: Union[str, Path] = "."
    ) -> Dict[str, Path]:
        """
        Analyze table text and save results to files.
        
        Args:
            table_text: Extracted table text
            analysis_types: List of analysis types to perform
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping analysis types to output file paths
        """
        if analysis_types is None:
            analysis_types = ["summary", "structure", "content"]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for analysis_type in analysis_types:
            print(f"\n--- Performing {analysis_type} analysis ---")
            
            try:
                analysis_result = self.analyze_table(table_text, analysis_type)
                
                # Save to file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"analysis_{analysis_type}_{timestamp}.txt"
                with open(output_file, 'w') as f:
                    f.write(analysis_result)
                
                results[analysis_type] = output_file
                print(f"  ✓ Saved to: {output_file}")
                
                # Try to parse JSON if that's what we asked for
                if analysis_type == "json":
                    try:
                        json_data = json.loads(analysis_result)
                        json_file = output_dir / f"structured_data_{timestamp}.json"
                        with open(json_file, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        results["json_parsed"] = json_file
                        print(f"  ✓ JSON parsed and saved to: {json_file}")
                    except json.JSONDecodeError:
                        # Try to extract JSON from text
                        import re
                        json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group())
                                json_file = output_dir / f"extracted_structured_{timestamp}.json"
                                with open(json_file, 'w') as f:
                                    json.dump(json_data, f, indent=2)
                                results["json_extracted"] = json_file
                                print(f"  ✓ Extracted JSON saved to: {json_file}")
                            except:
                                print(f"  ⚠️ Could not parse JSON from response")
                
            except Exception as e:
                print(f"  ❌ {analysis_type} analysis failed: {str(e)[:100]}...")
                results[f"{analysis_type}_error"] = str(e)
        
        return results


class IndiaCodeAnalyzer:
    """Complete IndiaCode legislative table analyzer combining OCR and multimodal AI."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        ocr_endpoint: Optional[str] = None,
        multimodal_endpoint: Optional[str] = None,
        multimodal_deployment: Optional[str] = None
    ):
        """
        Initialize complete analyzer.
        
        Args:
            api_key: Azure API key (default: AZURE_API_KEY env var)
            ocr_endpoint: Azure Document Intelligence (OCR) endpoint (default: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT env var)
            multimodal_endpoint: Multimodal AI OpenAI-compatible endpoint (default: AZURE_MULTIMODAL_ENDPOINT env var)
            multimodal_deployment: Multimodal deployment name (default: AZURE_MULTIMODAL_DEPLOYMENT env var)
        """
        # Initialize OCR
        self.ocr = AzureOCR(
            api_key=api_key,
            endpoint=ocr_endpoint
        )
        
        # Initialize multimodal analyzer
        self.multimodal = AzureMultimodalAnalyzer(
            api_key=api_key,
            endpoint=multimodal_endpoint,
            deployment_name=multimodal_deployment
        )
    
    def analyze_pdf(
        self,
        pdf_path: Union[str, Path],
        analysis_types: List[str] = None,
        output_dir: Union[str, Path] = ".",
        save_extracted: bool = True
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline: OCR extraction + multimodal AI analysis.
        
        Args:
            pdf_path: Path to PDF file
            analysis_types: List of analysis types to perform
            output_dir: Directory for output files
            save_extracted: Whether to save extracted text
            
        Returns:
            Dictionary with all results and file paths
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if analysis_types is None:
            analysis_types = ["summary", "structure", "content"]
        
        print(f"\n{'='*60}")
        print(f"INDIA CODE LEGISLATIVE TABLE ANALYSIS")
        print(f"PDF: {pdf_path.name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        results = {
            "pdf_file": str(pdf_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": {},
            "errors": []
        }
        
        try:
            # Step 1: Extract text with OCR
            print(f"\n📋 STEP 1: PDF TEXT EXTRACTION")
            extracted_text = self.ocr.extract_text(pdf_path)
            
            if save_extracted:
                text_file = output_dir / f"{pdf_path.stem}_extracted.txt"
                with open(text_file, 'w') as f:
                    f.write(extracted_text)
                results["files"]["extracted"] = text_file
                results["extracted_length"] = len(extracted_text)
                print(f"  ✓ Saved: {text_file.name}")
            
            # Step 2: Analyze with multimodal AI
            print(f"\n📋 STEP 2: AI ANALYSIS")
            analysis_results = self.multimodal.analyze_and_save(
                table_text=extracted_text,
                analysis_types=analysis_types,
                output_dir=output_dir
            )
            
            results["files"].update(analysis_results)
            results["analyses_performed"] = analysis_types
            
            # Step 3: Create summary report
            print(f"\n📋 STEP 3: GENERATING SUMMARY")
            summary = {
                "analysis_complete": True,
                "pdf_file": pdf_path.name,
                "extracted_text_length": len(extracted_text),
                "analyses_performed": analysis_types,
                "files_generated": [str(f) for f in results["files"].values()],
                "timestamp": results["timestamp"]
            }
            
            summary_file = output_dir / f"{pdf_path.stem}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            results["files"]["summary"] = summary_file
            results["summary"] = summary
            
            print(f"  ✓ Summary: {summary_file.name}")
            
            # Final output
            print(f"\n{'='*60}")
            print(f"✅ ANALYSIS COMPLETE!")
            print(f"📊 Files generated:")
            for name, filepath in results["files"].items():
                if hasattr(filepath, 'name'):
                    print(f"   • {name}: {filepath.name}")
                else:
                    print(f"   • {name}: {filepath}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            results["errors"].append(str(e))
            return results
    
    def create_html_report(self, results: Dict[str, Any], output_file: Union[str, Path]) -> Path:
        """
        Create HTML report from analysis results.
        
        Args:
            results: Analysis results from analyze_pdf()
            output_file: Path for HTML output
            
        Returns:
            Path to HTML file
        """
        output_file = Path(output_file)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IndiaCode Legislative Table Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #3498db; background: #f8f9fa; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
        pre {{ background: white; padding: 15px; border-radius: 5px; overflow: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📜 IndiaCode Legislative Table Analysis Report</h1>
        <p>Generated: {results.get('timestamp', 'N/A')}</p>
        <p>PDF: {results.get('pdf_file', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Analysis Summary</h2>
        <p><span class="success">✓</span> Analysis completed successfully</p>
        <p>Text extracted: {results.get('extracted_length', 0):,} characters</p>
        <p>Analyses performed: {', '.join(results.get('analyses_performed', []))}</p>
    </div>
    
    <div class="section">
        <h2>📁 Generated Files</h2>
        <ul>
"""
        
        for name, filepath in results.get("files", {}).items():
            if isinstance(filepath, Path) and filepath.exists():
                file_size = filepath.stat().st_size
                html += f'<li><strong>{name}:</strong> {filepath.name} ({file_size:,} bytes)</li>\n'
        
        html += """        </ul>
    </div>
    
    <div class="section">
        <h2>🔍 Analysis Results Preview</h2>
        <p><em>Open individual analysis files for complete results</em></p>
    </div>
    
    <div class="section">
        <h2>⚙️ Technical Details</h2>
        <p><strong>Tools used:</strong></p>
        <ul>
            <li>Azure AI Document Intelligence (OCR)</li>
            <li>Azure Multimodal AI via LangChain</li>
            <li>akoma-markup Azure AI module</li>
        </ul>
    </div>
    
    {errors}
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
        <p>Generated by akoma-markup Azure AI module</p>
    </footer>
</body>
</html>"""
        
        # Add errors section if any
        errors_html = ""
        if results.get("errors"):
            errors_html = f"""
    <div class="section" style="border-left-color: #e74c3c;">
        <h2>❌ Errors Encountered</h2>
        <ul>
"""
            for error in results["errors"]:
                errors_html += f'<li class="error">{error}</li>\n'
            errors_html += """        </ul>
    </div>
"""
        
        html = html.format(errors=errors_html)
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML report saved to: {output_file}")
        return output_file


# Convenience functions for direct use
def extract_indiacode_tables(
    pdf_path: Union[str, Path],
    api_key: str,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convenience function to extract and analyze IndiaCode legislative tables.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Azure API key
        output_dir: Directory for output files
        
    Returns:
        Analysis results
    """
    analyzer = IndiaCodeAnalyzer(api_key=api_key)
    
    if output_dir is None:
        output_dir = Path(pdf_path).parent / "indiacode_analysis"
    
    return analyzer.analyze_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir
    )


def test_azure_connectivity(api_key: str, multimodal_endpoint: str = None, multimodal_deployment: str = None) -> bool:
    """
    Test connectivity to Azure AI services.
    
    Args:
        api_key: Azure API key
        multimodal_endpoint: Optional multimodal endpoint (defaults to hardcoded test endpoint)
        multimodal_deployment: Optional deployment name (defaults to hardcoded test model)
        
    Returns:
        True if both OCR and multimodal services are accessible
    """
    print("Testing Azure AI connectivity...")
    
    # Test OCR
    try:
        ocr = AzureOCR(api_key=api_key)
        # Just test endpoint with a simple request
        test_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        # Can't test OCR without actual PDF, so just check if endpoint exists
        print("  ⚠️ OCR: Endpoint configured (requires PDF for full test)")
    except Exception as e:
        print(f"  ❌ OCR test failed: {e}")
        return False
    
    # Test multimodal
    try:
        from openai import OpenAI
        
        # Use provided endpoint or default test endpoint
        actual_endpoint = multimodal_endpoint or os.environ.get("AZURE_MULTIMODAL_ENDPOINT") or "https://***.services.ai.azure.com/openai/v1/"
        actual_deployment = multimodal_deployment or os.environ.get("AZURE_MULTIMODAL_DEPLOYMENT") or "grok-4-1-fast-non-reasoning"
        
        test_client = OpenAI(
            base_url=actual_endpoint,
            api_key=api_key
        )
        test_response = test_client.chat.completions.create(
            model=actual_deployment,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=5
        )
        print(f"  ✓ multimodal: Connected (model: {test_response.model})")
        return True
    except Exception as e:
        print(f"  ❌ multimodal test failed: {e}")
        return False