from typing import Any

import networkx as nx
import numpy as np

from graphrag.llm import LLM
from graphrag.model import *
from graphrag.prompts.query.extraction import ENTITY_EXTRACTION
from graphrag.utils.embedding import get_embedding
from graphrag.utils.retrieval import get_cos_sim_matrix, retrieve
from graphrag.utils.transform import str2json


def get_seed_entities(query: str, llm: LLM, entities: list[Entity], threshold: float = 0.65) -> set[int]:
    """Identify a set of "seed entities"
    Seed entities are the initial entities that are chosen to be highly relevant to the original query.
    """
    name2id = {entity.name: entity.id for entity in entities}
    seed_ents = set()
    # Extract entities from the query
    response = llm.single_turn(ENTITY_EXTRACTION.format(input_text=query))
    extracted_entities = str2json(response)
    remains = []
    for entity in extracted_entities:
        if entity in name2id:
            seed_ents.add(name2id[entity])
        else:
            remains.append(entity)  # Subsequent batch process
    if remains:
        indices, _ = retrieve(
            query=np.array(get_embedding(remains)),
            target=np.array([entity.name_embedding for entity in entities]),
            threshold=threshold,
        )
        seed_ents.update(entities[id].id for ent_ids in indices for id in ent_ids)
    return seed_ents

def retrieve_text_units(query: list[str], text_units: list[TextUnit], **kwds) -> list[TextUnit]:
    indices, _ = retrieve(
        np.array(get_embedding(query)),
        np.array([text_unit.embedding for text_unit in text_units]),
        **kwds
    )
    return [text_units[text_unit_id] for text_unit_id in indices[0]]

def max_avg_weight_path(graph: nx.Graph, s: Any, t: Any, epsilon=1e-1):
    """Using binary strategy to find the path with the maximum average weight from s to t.
    """
    def _get_path(alpha: float):
        # Build a weighted graph with weights greater than alpha.
        modified_g = nx.Graph()
        for u, v, weight in graph.edges.data("weight"):
            if weight >= alpha:
                modified_g.add_edge(u, v, weight=weight)
        if s not in modified_g.nodes() or t not in modified_g.nodes():
            return None
        # Determine whether there is a shortest path under the current weight
        try:
            path = nx.dijkstra_path(modified_g, s, t)
            return path
        except nx.NetworkXNoPath:
            return None

    min, max = 0, 1
    best_path = None
    while max - min > epsilon:
        mid = (min + max) / 2
        path = _get_path(mid)
        if path:
            best_path = path
            min = mid
        else:
            max = mid
    return best_path[1:-1]

def retrieve_subgraph(query: str, graph: nx.Graph, seed_nodes: set[int], relations: list[Relation], threshold: float=0.65) -> nx.Graph:
    entities = set(seed_nodes)
    weights = get_cos_sim_matrix(
        np.array(get_embedding(query)),
        np.array([relation.embedding for relation in relations]),
    )[0]
    c = 0  # TODO: del later
    for i in range(len(weights)):
        if weights[i] >= threshold:
            entities.add(relations[i].source)
            entities.add(relations[i].target)
            c += 1  # TODO: del later
    subgraph = graph.subgraph(entities)
    # connected_components = list(nx.connected_components(subgraph))
    # # If it is already a connected graph, there is no need to connect it.
    # if len(connected_components) > 1:
    #     # Build a weighted graph.
    #     wg = nx.Graph()
    #     wg.add_nodes_from(graph.nodes())
    #     for i, (u, v) in enumerate(graph.edges()):
    #         wg.add_edge(u, v, weight=weights[i])
    #     merged_nodes = [node for component in connected_components for node in component]
    #     new_nodes = []
    #     for i, component in enumerate(connected_components):
    #         # Merge the connected graph into one node.
    #         new_node = f"new_node{i}"
    #         new_nodes.append(new_node)
    #         wg.add_node(new_node)
    #         for node in component:
    #             for neighbor in wg.neighbors(node):
    #                 if neighbor not in merged_nodes:
    #                     wg.add_edge(new_node, neighbor, weight=wg.edges[node, neighbor]["weight"])
    #         # Delete nodes from the original connected graph.
    #         for node in component:
    #             wg.remove_node(node)
    #     for i in range(len(new_nodes)):
    #         for j in range(i + 1, len(new_nodes)):
    #             s = new_nodes[i]
    #             t = new_nodes[j]
    #             try:
    #                 nx.dijkstra_path(wg, s, t)
    #                 entities.update(max_avg_weight_path(wg, s, t))
    #             except:
    #                 pass
    #     subgraph = graph.subgraph(list(entities))
    return subgraph