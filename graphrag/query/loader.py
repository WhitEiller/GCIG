import os
from typing import Any

import networkx as nx
import pandas as pd

from graphrag.model import Entity, Relation


def load_parquet(path) -> list[dict[str, Any]]:
    records = pd.read_parquet(path, engine="pyarrow").to_dict(orient="records")
    for record in records:
        for k, v in record.items():
            if isinstance(v, pd.Series):
                record[k] = v.tolist()
    return records

def load_graph(graph_dir: str) -> tuple[nx.Graph, list[Entity], list[Relation]]:
    # Load data from parquet
    entities = load_parquet(os.path.join(graph_dir, "entities.parquet"))
    relations = load_parquet(os.path.join(graph_dir, "relations.parquet"))

    entities = [Entity.from_dict(entity) for entity in entities]
    relations = [Relation.from_dict(relation) for relation in relations]

    # Construct graph
    graph = nx.Graph()
    for entity in entities:
        graph.add_node(entity.id,name=entity.name,description=entity.description)

    for relation in relations:
        graph.add_edge(relation.source, relation.target, id=relation.id,description = relation.description)

    return graph, entities, relations
