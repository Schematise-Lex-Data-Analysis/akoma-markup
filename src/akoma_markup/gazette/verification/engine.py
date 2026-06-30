"""LLM verification engine for Gazette refinement output."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models import BaseChatModel

from ..verification_prompts import (
    VERIFICATION_SYSTEM_PROMPT_TEMPLATE,
    VERIFICATION_HUMAN_TEMPLATE,
    VERIFICATION_CHAIN_SYSTEM_PROMPT,
    VERIFICATION_CHAIN_HUMAN_TEMPLATE,
)
from .checkpoint import VerificationCheckpointManager
from .processor import ContentProcessorChain

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of a section verification."""
    
    section_num: str
    original_content: str
    verified_content: str
    confidence_score: float
    issues_fixed: List[str]
    section_numbers_present: List[str]
    recommendations: List[str]
    verification_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_llm_response(cls, section_num: str, original_content: str, 
                         llm_response: Dict[str, Any]) -> "VerificationResult":
        """Create VerificationResult from LLM JSON response."""
        return cls(
            section_num=section_num,
            original_content=original_content,
            verified_content=llm_response.get("verified_content", original_content),
            confidence_score=llm_response.get("confidence_score", 0.0),
            issues_fixed=llm_response.get("issues_fixed", []),
            section_numbers_present=llm_response.get("section_numbers_present", []),
            recommendations=llm_response.get("recommendations", []),
            verification_metadata=llm_response,
        )


class LLMVerificationEngine:
    """Orchestrates LLM-powered verification of Gazette refinement output."""
    
    def __init__(
        self,
        llm: BaseChatModel,
        verification_level: str = "moderate",
        context_window: int = 3,
        remove_brackets: bool = True,
        batch_size: int = 5,
        checkpoint_manager: Optional[VerificationCheckpointManager] = None,
    ):
        """Initialize verification engine.
        
        Args:
            llm: LangChain chat model for verification
            verification_level: "strict", "moderate", or "light"
            context_window: Number of surrounding sections to include for context
            remove_brackets: Whether to remove unnecessary square brackets
            batch_size: Number of sections to process in parallel
            checkpoint_manager: Optional checkpoint manager for resume capability
        """
        self.llm = llm
        self.verification_level = verification_level
        self.context_window = context_window
        self.remove_brackets = remove_brackets
        self.batch_size = batch_size
        self.checkpoint_manager = checkpoint_manager or VerificationCheckpointManager()
        
        # Build verification chain
        self._build_verification_chain()
        
        # Initialize processor chain
        self.processor_chain = ContentProcessorChain()
        
        logger.info(
            f"Initialized LLMVerificationEngine: "
            f"level={verification_level}, "
            f"context={context_window}, "
            f"batch={batch_size}"
        )
    
    def _build_verification_chain(self):
        """Build the LangChain verification pipeline."""
        from langchain_core.prompts import ChatPromptTemplate
        
        system_prompt = VERIFICATION_SYSTEM_PROMPT_TEMPLATE
        human_prompt = VERIFICATION_HUMAN_TEMPLATE
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])
        
        parser = JsonOutputParser()
        self.verification_chain = prompt | self.llm | parser
        
        # Also build comprehensive chain for combined processing
        comprehensive_prompt = ChatPromptTemplate.from_messages([
            ("system", VERIFICATION_CHAIN_SYSTEM_PROMPT),
            ("human", VERIFICATION_CHAIN_HUMAN_TEMPLATE),
        ])
        self.comprehensive_chain = comprehensive_prompt | self.llm | parser
    
    async def verify_section(
        self,
        section_num: str,
        content: str,
        heading: str = "",
        context_sections: Optional[List[Tuple[str, str, str]]] = None,
    ) -> VerificationResult:
        """Verify a single section using LLM.
        
        Args:
            section_num: Section identifier
            content: Section content to verify
            heading: Section heading (if available)
            context_sections: List of (section_num, heading, content) for surrounding sections
            
        Returns:
            VerificationResult with cleaned content and metadata
        """
        # Prepare context
        context_str = self._format_context(context_sections or [])
        
        # Prepare input for LLM
        llm_input = {
            "section_num": section_num,
            "heading": heading or "",
            "verification_level": self.verification_level,
            "context_window": self.context_window,
            "context": context_str,
            "content": content,
        }
        
        try:
            # Get LLM verification
            if self.verification_level == "strict":
                llm_response = await self.comprehensive_chain.ainvoke(llm_input)
            else:
                llm_response = await self.verification_chain.ainvoke(llm_input)
            
            # Apply post-processing if needed
            verified_content = llm_response.get("verified_content", content)
            if self.remove_brackets:
                verified_content = self.processor_chain.process(verified_content)
            
            # Create result
            result = VerificationResult.from_llm_response(
                section_num=section_num,
                original_content=content,
                llm_response=llm_response,
            )
            
            # Update with post-processed content if changed
            if verified_content != result.verified_content:
                result.verified_content = verified_content
                result.issues_fixed.append("applied_bracket_cleanup")
            
            logger.debug(f"Verified section {section_num}: confidence={result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to verify section {section_num}: {e}")
            # Return fallback result
            return VerificationResult(
                section_num=section_num,
                original_content=content,
                verified_content=content,
                confidence_score=0.0,
                issues_fixed=["verification_failed"],
                section_numbers_present=[],
                recommendations=[f"Verification error: {str(e)}"],
            )
    
    def _format_context(self, context_sections: List[Tuple[str, str, str]]) -> str:
        """Format surrounding sections for context."""
        if not context_sections:
            return "(no context available)"
        
        lines = []
        for section_num, heading, content in context_sections:
            preview = content[:200] + "..." if len(content) > 200 else content
            lines.append(f"--- Section {section_num} ({heading}) ---")
            lines.append(preview)
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_context_for_row(
        self,
        df: pd.DataFrame,
        idx: int,
        context_window: int = None,
    ) -> List[Tuple[str, str, str]]:
        """Get surrounding sections for a specific row index."""
        if context_window is None:
            context_window = self.context_window
        
        context_sections = []
        
        # Get rows before current
        start_idx = max(0, idx - context_window)
        for i in range(start_idx, idx):
            if i < len(df):
                row = df.iloc[i]
                section_num = str(row.get("section", "")).strip()
                heading = str(row.get("heading", "")).strip()
                content = str(row.get("content", "")).strip()
                if content:
                    context_sections.append((section_num, heading, content))
        
        # Get rows after current
        end_idx = min(len(df), idx + context_window + 1)
        for i in range(idx + 1, end_idx):
            if i < len(df):
                row = df.iloc[i]
                section_num = str(row.get("section", "")).strip()
                heading = str(row.get("heading", "")).strip()
                content = str(row.get("content", "")).strip()
                if content:
                    context_sections.append((section_num, heading, content))
        
        return context_sections
    
    async def verify_tsv(
        self,
        tsv_path: Union[str, Path],
        output_path: Union[str, Path],
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Verify all sections in a TSV file.
        
        Args:
            tsv_path: Path to input TSV file
            output_path: Path for output TSV file
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary with verification summary statistics
        """
        tsv_path = Path(tsv_path)
        output_path = Path(output_path)
        
        logger.info(f"Starting TSV verification: {tsv_path} -> {output_path}")
        
        # Load TSV
        try:
            df = pd.read_csv(tsv_path, sep="\t", dtype=str)
            required_cols = {"section", "heading", "content"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        except Exception as e:
            logger.error(f"Failed to load TSV {tsv_path}: {e}")
            raise
        
        # Check for checkpoint
        checkpoint_data = None
        if resume:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(tsv_path, output_path)
        
        # Prepare results DataFrame
        if checkpoint_data and "results_df" in checkpoint_data:
            results_df = checkpoint_data["results_df"]
            start_idx = checkpoint_data.get("processed_count", 0)
            logger.info(f"Resuming from checkpoint: {start_idx} sections processed")
        else:
            # Initialize results DataFrame
            results_df = df.copy()
            results_df["verification_confidence"] = ""
            results_df["issues_fixed"] = ""
            results_df["verified_content"] = ""
            start_idx = 0
        
        # Process sections in batches
        total_sections = len(df)
        processed_count = start_idx
        failed_count = 0
        
        for batch_start in range(start_idx, total_sections, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_sections)
            batch_indices = list(range(batch_start, batch_end))
            
            logger.info(f"Processing batch {batch_start+1}-{batch_end} of {total_sections}")
            
            # Prepare batch tasks
            tasks = []
            for idx in batch_indices:
                row = df.iloc[idx]
                section_num = str(row["section"]).strip()
                heading = str(row.get("heading", "")).strip()
                content = str(row["content"]).strip()
                
                # Skip if already processed in results_df
                if not pd.isna(results_df.iloc[idx]["verified_content"]):
                    continue
                
                # Get context
                context_sections = self._get_context_for_row(df, idx)
                
                # Create verification task
                task = self.verify_section(
                    section_num=section_num,
                    content=content,
                    heading=heading,
                    context_sections=context_sections,
                )
                tasks.append((idx, task))
            
            # Execute batch
            if tasks:
                verification_tasks = [task for _, task in tasks]
                results = await asyncio.gather(*verification_tasks, return_exceptions=True)
                
                # Process results
                for (idx, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to verify section at index {idx}: {result}")
                        # Store original content as fallback
                        results_df.at[idx, "verified_content"] = df.iloc[idx]["content"]
                        results_df.at[idx, "verification_confidence"] = "0.0"
                        results_df.at[idx, "issues_fixed"] = "verification_error"
                        failed_count += 1
                    else:
                        # Store verification result
                        results_df.at[idx, "verified_content"] = result.verified_content
                        results_df.at[idx, "verification_confidence"] = str(result.confidence_score)
                        results_df.at[idx, "issues_fixed"] = "; ".join(result.issues_fixed)
                    
                    processed_count += 1
                
                # Save checkpoint
                checkpoint_data = {
                    "processed_count": processed_count,
                    "results_df": results_df,
                    "failed_count": failed_count,
                }
                self.checkpoint_manager.save_checkpoint(
                    tsv_path, output_path, checkpoint_data
                )
            
            # Progress update
            progress_pct = (processed_count / total_sections) * 100
            logger.info(f"Progress: {processed_count}/{total_sections} ({progress_pct:.1f}%)")
        
        # Save final output
        try:
            # Update content column with verified content
            results_df["content"] = results_df["verified_content"]
            
            # Drop helper columns for output
            output_df = results_df.drop(columns=["verified_content", "verification_confidence", "issues_fixed"], errors="ignore")
            output_df.to_csv(output_path, sep="\t", index=False)
            
            logger.info(f"Saved verified TSV to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output TSV: {e}")
            raise
        
        # Generate summary
        confidence_scores = []
        for conf_str in results_df["verification_confidence"]:
            try:
                confidence_scores.append(float(conf_str))
            except (ValueError, TypeError):
                confidence_scores.append(0.0)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        success_rate = (processed_count - failed_count) / processed_count if processed_count > 0 else 0.0
        
        summary = {
            "input_path": str(tsv_path),
            "output_path": str(output_path),
            "total_sections": total_sections,
            "verified_sections": processed_count,
            "failed_sections": failed_count,
            "avg_confidence": avg_confidence,
            "success_rate": success_rate,
            "verification_level": self.verification_level,
            "context_window": self.context_window,
        }
        
        logger.info(
            f"Verification complete: "
            f"{processed_count-failed_count}/{processed_count} successful, "
            f"avg confidence={avg_confidence:.2f}"
        )
        
        # Clean up checkpoint
        self.checkpoint_manager.clear_checkpoint(tsv_path, output_path)
        
        return summary