"""
ToG (Think-on-Graph) Reasoning Adapter

This module provides a simplified adapter for integrating ToG's multi-hop reasoning
capabilities with the graphrag framework.

Reference:
    Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Shum, H. Y., & Guo, J. (2024).
    Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph.
    In International Conference on Learning Representations (ICLR).
"""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx

from graphrag.llm import LLM
from graphrag.model import Entity, Relation
from graphrag.prompts.query.tog import (
    EXTRACT_RELATION_PROMPT,
    SCORE_ENTITY_CANDIDATES_PROMPT,
    ANSWER_PROMPT,
    PROMPT_EVALUATE,
    COT_PROMPT
)
from graphrag.utils.tog_utils import (
    clean_relations,
    clean_scores,
    del_unknown_entity
)


class ToGReasoner:
    """
    Think-on-Graph reasoning engine for multi-hop question answering over knowledge graphs.

    This adapter provides a simplified interface to ToG's iterative graph exploration
    and reasoning capabilities, integrated with graphrag's data models and LLM interface.
    """

    def __init__(
        self,
        llm: LLM,
        depth: int = 3,
        width: int = 3,
        exploration_temperature: float = 0.4,
        reasoning_temperature: float = 0.0,
        max_tokens: int = 256
    ):
        """
        Initialize ToG reasoner.

        Args:
            llm: Language model instance (graphrag.llm.LLM)
            depth: Maximum depth for graph exploration (default: 3)
            width: Number of top candidates to keep at each step (default: 3)
            exploration_temperature: Temperature for exploration phase (default: 0.4)
            reasoning_temperature: Temperature for reasoning phase (default: 0.0)
            max_tokens: Maximum tokens for LLM generation (default: 256)
        """
        self.llm = llm
        self.depth = depth
        self.width = width
        self.exploration_temp = exploration_temperature
        self.reasoning_temp = reasoning_temperature
        self.max_tokens = max_tokens

    def reason(
        self,
        query: str,
        graph: nx.Graph,
        entities: List[Entity],
        relations: List[Relation],
        seed_entities: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Perform multi-hop reasoning over the knowledge graph to answer a query.

        Args:
            query: Natural language question
            graph: NetworkX graph containing the knowledge graph
            entities: List of Entity objects
            relations: List of Relation objects
            seed_entities: Optional list of seed entity names to start exploration

        Returns:
            Tuple of (reasoning_chain, answer)
            - reasoning_chain: List of reasoning steps with entities and relations
            - answer: Final answer to the query
        """
        # Step 1: Identify seed entities if not provided
        if seed_entities is None:
            seed_entities = self._extract_seed_entities(query, entities)

        # Step 2: Iterative graph exploration
        reasoning_chain = []
        current_entities = seed_entities

        for depth_level in range(self.depth):
            # Step 2a: Relation exploration
            candidate_relations = self._explore_relations(
                query, current_entities, graph, depth_level
            )

            if not candidate_relations:
                break

            # Step 2b: Entity search
            next_entities = self._search_entities(
                query, candidate_relations, graph
            )

            if not next_entities:
                break

            # Step 2c: Record reasoning step
            reasoning_chain.append({
                "depth": depth_level,
                "entities": current_entities,
                "relations": candidate_relations,
                "next_entities": next_entities
            })

            # Step 2d: Evaluate if sufficient information gathered
            triplets = self._build_triplets(reasoning_chain)
            is_sufficient = self._evaluate_sufficiency(query, triplets)

            if is_sufficient:
                break

            current_entities = next_entities

        # Step 3: Generate final answer
        triplets = self._build_triplets(reasoning_chain)
        answer = self._generate_answer(query, triplets)

        return reasoning_chain, answer

    def _extract_seed_entities(self, query: str, entities: List[Entity]) -> List[str]:
        """Extract seed entities from the query."""
        # Simple implementation: find entities mentioned in query
        seed_entities = []
        query_lower = query.lower()

        for entity in entities:
            if entity.name.lower() in query_lower:
                seed_entities.append(entity.name)

        # If no entities found, return empty list
        # In production, this could use more sophisticated entity linking
        return seed_entities[:self.width]

    def _explore_relations(
        self,
        query: str,
        current_entities: List[str],
        graph: nx.Graph,
        depth_level: int
    ) -> List[Dict[str, Any]]:
        """
        Explore and prune relations for current entities.

        Returns:
            List of relation dictionaries with scores
        """
        all_relations = []

        for entity_name in current_entities:
            if entity_name not in graph.nodes:
                continue

            # Get all relations (edges) for this entity
            entity_relations = []
            for neighbor in graph.neighbors(entity_name):
                edge_data = graph.get_edge_data(entity_name, neighbor)
                if edge_data and 'relations' in edge_data:
                    entity_relations.extend(edge_data['relations'])

            if not entity_relations:
                continue

            # Use LLM to prune relations
            # Format: semicolon-separated list
            relations_str = "; ".join(set(entity_relations))

            prompt = EXTRACT_RELATION_PROMPT % (self.width, self.width)
            prompt += f"{query}\nTopic Entity: {entity_name}\nRelations: {relations_str}\nA:"

            llm_output = self.llm.single_turn(prompt)

            # Parse LLM output
            success, parsed_relations = clean_relations(llm_output, entity_name, entity_relations)

            if success:
                all_relations.extend(parsed_relations)

        # Sort by score and return top-width
        all_relations.sort(key=lambda x: x['score'], reverse=True)
        return all_relations[:self.width]

    def _search_entities(
        self,
        query: str,
        relations: List[Dict[str, Any]],
        graph: nx.Graph
    ) -> List[str]:
        """
        Search for candidate entities given relations.

        Returns:
            List of entity names
        """
        candidate_entities = set()

        for rel_dict in relations:
            entity_name = rel_dict['entity']
            relation = rel_dict['relation']

            if entity_name not in graph.nodes:
                continue

            # Find neighbors connected via this relation
            for neighbor in graph.neighbors(entity_name):
                edge_data = graph.get_edge_data(entity_name, neighbor)
                if edge_data and 'relations' in edge_data:
                    if relation in edge_data['relations']:
                        candidate_entities.add(neighbor)

        return list(candidate_entities)[:self.width]

    def _build_triplets(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """
        Build knowledge graph triplets string from reasoning chain.

        Returns:
            Formatted triplets string
        """
        triplets = []

        for step in reasoning_chain:
            entities = step['entities']
            relations = step['relations']
            next_entities = step['next_entities']

            for rel_dict in relations:
                for next_ent in next_entities:
                    triplet = f"{rel_dict['entity']}, {rel_dict['relation']}, {next_ent}"
                    triplets.append(triplet)

        return "\n".join(triplets)

    def _evaluate_sufficiency(self, query: str, triplets: str) -> bool:
        """
        Evaluate if current knowledge triplets are sufficient to answer the query.

        Returns:
            True if sufficient, False otherwise
        """
        prompt = PROMPT_EVALUATE + f"Q: {query}\nKnowledge Triplets: {triplets}\nA:"

        llm_output = self.llm.single_turn(prompt)

        # Check if LLM says "Yes"
        return "{Yes}" in llm_output or "{yes}" in llm_output

    def _generate_answer(self, query: str, triplets: str) -> str:
        """
        Generate final answer given query and knowledge triplets.

        Returns:
            Answer string
        """
        prompt = ANSWER_PROMPT.format(query)
        prompt = prompt.replace("Q: {}", f"Q: {query}")
        prompt += f"\nKnowledge Triplets: {triplets}\nA:"

        answer = self.llm.single_turn(prompt)

        return answer


class ToGAdapter:
    """
    Convenience wrapper for ToG reasoning with graphrag data structures.
    """

    @staticmethod
    def reason_with_graphrag(
        query: str,
        graph: nx.Graph,
        entities: List[Entity],
        relations: List[Relation],
        llm: LLM,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Perform ToG reasoning using graphrag data structures.

        Args:
            query: Natural language question
            graph: NetworkX graph
            entities: List of Entity objects
            relations: List of Relation objects
            llm: LLM instance
            **kwargs: Additional arguments for ToGReasoner

        Returns:
            Tuple of (reasoning_chain, answer)
        """
        reasoner = ToGReasoner(llm=llm, **kwargs)
        return reasoner.reason(query, graph, entities, relations)


__all__ = ["ToGReasoner", "ToGAdapter"]

