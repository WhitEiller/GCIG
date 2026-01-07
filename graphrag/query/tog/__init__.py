"""
ToG (Think-on-Graph) Integration Module

This module provides ToG's multi-hop reasoning capabilities
integrated with the graphrag framework.

Reference:
    Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Shum, H. Y., & Guo, J. (2024).
    Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph.
    In International Conference on Learning Representations (ICLR).
"""

from .tog_adapter import ToGReasoner, ToGAdapter

__all__ = ["ToGReasoner", "ToGAdapter"]
