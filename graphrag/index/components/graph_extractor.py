from typing import Any

import networkx as nx

from graphrag.llm.base_model import LLM
from graphrag.prompts.index.extraction import (CONTINUE_PROMPT,
                                               GRAPH_EXTRACTION_PROMPT,
                                               LOOP_PROMPT)
from graphrag.utils.transform import str2json


class GraphExtractor:
    def __init__(self,
                 llm: LLM,
                 entity_types: list[str] | None = None,
                 prompt: str | None = None,
                 max_gleanings: int = 1):
        self._llm = llm
        self._extraction_prompt = prompt or GRAPH_EXTRACTION_PROMPT
        self._entity_types = (
            entity_types
            if entity_types
            else ["PERSON", "ORGANIZATION", "PRODUCT", "LEGALTERM", "CONDITION"]
        )
        self._max_gleanings = max_gleanings

    def _extract(self, text: str) -> dict[str, Any]:
        self._llm.reset()
        results = []
        response = self._llm.multi_turn(
            self._extraction_prompt.format(input_text=text, entity_types=self._entity_types)
        )
        results.append(response)

        for _ in range(self._max_gleanings - 1):
            response = self._llm.multi_turn(CONTINUE_PROMPT)
            results.append(response)
            
            # determine there further extraction is necessary
            response = self._llm.multi_turn(LOOP_PROMPT)
            if response != "YES":
                break
        return results

    def _process_results(self, entities: list[Any], relations: list[Any]) -> nx.Graph:
        graph = nx.Graph()
        for entity in entities:
            # Nodes with the same name and type are regarded as identical nodes
            if entity["name"] in graph.nodes() and entity["type"] == graph.nodes[entity["name"]]["type"]:
                graph.nodes[entity["name"]]["description"].append(entity["description"])
            else:
                graph.add_node(
                    entity["name"],
                    type=entity["type"],
                    description=[entity["description"]]
                )

        for relation in relations:
            source = relation["source"]
            target = relation["target"]
            if source not in graph.nodes():
                graph.add_node(
                    source,
                    type="",
                    description=[],
                )
            if target not in graph.nodes():
                graph.add_node(
                    target,
                    type="",
                    description=[],
                )
            if graph.has_edge(source, target):
                graph.edges[source, target]["relations"].append(relation["description"])
            else:
                graph.add_edge(
                    source,
                    target,
                    relations=[relation["description"]]
                )
        return graph

    def __call__(self, texts: str | list[str]) -> nx.Graph:
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            results.extend(self._extract(text))

        entities = []
        relations = []
        for result in results:
            graph = str2json(result)
            entities.extend(graph.get("entities", []))
            relations.extend(graph.get("relations", []))
        return self._process_results(entities, relations)