from typing import Any, Callable

import networkx as nx
import numpy as np

from graphrag.llm import LLM
from graphrag.model import Entity
from graphrag.prompts.index.alignment import ENTITY_ALIGNMENT_PROMPT
from graphrag.utils.retrieval import get_cos_sim_matrix
from graphrag.utils.transform import str2json


def similarity_align(entities: list[Entity], threshold: float = 0.75) -> list[list[Entity]]:
    embeddings = np.array([entity.desc_embedding for entity in entities])
    # Calculate the cosine similarity between entities
    sim_matrix = get_cos_sim_matrix(embeddings, embeddings)
    
    # Node similarity graph
    g = nx.Graph()
    # Filter out entity pairs with similarity higher than the threshold
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if sim_matrix[i][j] >= threshold:
                g.add_edge(i, j)
    
    # Calculate the connected graph and obtain the set of all similar entities.
    entity_sets = list(nx.connected_components(g))
    return [[entities[entity_id] for entity_id in entity_set] for entity_set in entity_sets]

def llm_align(entities: list[Entity], llm: LLM) -> list[list[Entity]]:
    for _ in range(5):
        response = llm.single_turn(ENTITY_ALIGNMENT_PROMPT.format(
            entities=[{"name": entity.name, "description": entity.description}
                    for entity in entities]
        ))
        try:
            entity_sets = str2json(response)
        except:
            continue
        break
    else:
        return []
    # Map entity_name to entity
    name2ent = {entity.name: entity for entity in entities}
    final_ent_sets = []
    for entity_set in entity_sets:
        final_ent_set = [name2ent[entity] for entity in entity_set if entity in name2ent]
        # Ignore set with only on entity
        if len(final_ent_set) > 1:
            final_ent_sets.append(final_ent_set)
    return final_ent_sets

def type_align(entities: list[Entity]) -> list[list[Entity]]:
    entity_sets: dict[str, list[Entity]] = {}
    for entity in entities:
        if entity.type is None:
            continue
        type = entity.type
        if type in entity_sets:
            entity_sets[type].append(entity)
        else:
            entity_sets[type] = [entity]
    return [entity_set for _, entity_set in entity_sets.items() if len(entity_set) > 1]


class Aligner:
    def __init__(self) -> None:
        self._step: list[tuple[Callable, dict[str, Any]]] = []
    
    def add(self, func: Callable, **params: dict[str, Any]):
        self._step.append((func, params))
    
    def run(self, entities: list[Entity]):
        # Use alignment methods sequentially
        entity_sets = [entities]
        for func, params in self._step:
            # There are no entity sets that require aligning
            if not entity_sets:
                break
            
            results = []
            for entity_set in entity_sets:
                result = func(entity_set, **params)
                if result:
                    results.extend(result)
            entity_sets = results
        
        # Merge entity node
        for entity_set in entity_sets:
            for entity in entity_set:
                # Add alias attribue to the entity
                # Entity ID indicating that it is the same entity as the `entity`
                entity.alias = [ent.id for ent in entity_set if ent != entity]