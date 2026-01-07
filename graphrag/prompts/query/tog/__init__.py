"""
ToG (Think-on-Graph) Prompt Templates

This module contains prompt templates extracted from the ToG (Think-on-Graph) algorithm
for deep reasoning on knowledge graphs.

Reference:
    Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Shum, H. Y., & Guo, J. (2024).
    Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph.
    In International Conference on Learning Representations (ICLR).
"""

from .relation_prompts import (
    EXTRACT_RELATION_PROMPT,
    EXTRACT_RELATION_PROMPT_WIKI
)

from .entity_prompts import (
    SCORE_ENTITY_CANDIDATES_PROMPT,
    SCORE_ENTITY_CANDIDATES_PROMPT_WIKI
)

from .reasoning_prompts import (
    ANSWER_PROMPT,
    ANSWER_PROMPT_WIKI,
    PROMPT_EVALUATE,
    PROMPT_EVALUATE_WIKI,
    COT_PROMPT,
    GENERATE_DIRECTLY
)

__all__ = [
    # Relation extraction prompts
    "EXTRACT_RELATION_PROMPT",
    "EXTRACT_RELATION_PROMPT_WIKI",

    # Entity scoring prompts
    "SCORE_ENTITY_CANDIDATES_PROMPT",
    "SCORE_ENTITY_CANDIDATES_PROMPT_WIKI",

    # Reasoning and answer generation prompts
    "ANSWER_PROMPT",
    "ANSWER_PROMPT_WIKI",
    "PROMPT_EVALUATE",
    "PROMPT_EVALUATE_WIKI",
    "COT_PROMPT",
    "GENERATE_DIRECTLY",
]
