"""Chain builders for Gazette verification."""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..verification_prompts import (
    BRACKET_CLEANUP_SYSTEM_PROMPT,
    DUPLICATE_DETECTION_SYSTEM_PROMPT,
    SECTION_NORMALIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT_TEMPLATE,
    VERIFICATION_HUMAN_TEMPLATE,
    VERIFICATION_CHAIN_SYSTEM_PROMPT,
    VERIFICATION_CHAIN_HUMAN_TEMPLATE,
)


def build_bracket_cleanup_chain(llm):
    """Build chain for bracket cleanup only."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", BRACKET_CLEANUP_SYSTEM_PROMPT),
        ("human", "{content}"),
    ])
    return prompt | llm


def build_duplicate_detection_chain(llm):
    """Build chain for duplicate detection."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", DUPLICATE_DETECTION_SYSTEM_PROMPT),
        ("human", "{content}"),
    ])
    parser = JsonOutputParser()
    return prompt | llm | parser


def build_section_normalization_chain(llm):
    """Build chain for section normalization."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SECTION_NORMALIZATION_SYSTEM_PROMPT),
        ("human", "{content}"),
    ])
    parser = JsonOutputParser()
    return prompt | llm | parser


def build_verification_chain(llm, verification_level="moderate"):
    """Build comprehensive verification chain.
    
    Args:
        llm: LangChain chat model
        verification_level: "strict", "moderate", or "light"
    
    Returns:
        LangChain chain for verification
    """
    if verification_level == "strict":
        system_prompt = VERIFICATION_CHAIN_SYSTEM_PROMPT
        human_prompt = VERIFICATION_CHAIN_HUMAN_TEMPLATE
    else:
        system_prompt = VERIFICATION_SYSTEM_PROMPT_TEMPLATE
        human_prompt = VERIFICATION_HUMAN_TEMPLATE
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    
    parser = JsonOutputParser()
    return prompt | llm | parser


def build_specialized_chain(llm, chain_type):
    """Build specialized chain based on type.
    
    Args:
        llm: LangChain chat model
        chain_type: "bracket_cleanup", "duplicate_detection", 
                   "section_normalization", or "full_verification"
    
    Returns:
        Appropriate LangChain chain
    """
    if chain_type == "bracket_cleanup":
        return build_bracket_cleanup_chain(llm)
    elif chain_type == "duplicate_detection":
        return build_duplicate_detection_chain(llm)
    elif chain_type == "section_normalization":
        return build_section_normalization_chain(llm)
    elif chain_type == "full_verification":
        return build_verification_chain(llm, verification_level="strict")
    else:
        raise ValueError(f"Unknown chain type: {chain_type}")