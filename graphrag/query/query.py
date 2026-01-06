from graphrag.llm import LLM
from graphrag.prompts.query.answer import ANSWER_PROMPT

from .retrieval import *
from collections import Counter

def get_context(texts, graph=None):
    context = []
    if graph:
        context.append("="*5 + "Entities" + "="*5)
        context.extend(f"{graph.nodes[node]['name']}: {graph.nodes[node]['description']}" for node in graph.nodes())

        context.append("="*5 + "Relationships" + "="*5)
        context.extend(f"{u}, {v}: {graph.edges[u, v]['description']}" for u, v in graph.edges())

    context.append("="*5 + "Sources" + "="*5)
    context.extend(texts)
    return "\n".join(context)

def generate(
    query: str,
    llm: LLM,
    threshold: dict[str, float],
    top_k: int,
    required_graph: bool,
    graph: nx.Graph,
    entities: list[Entity],
    relations: list[Relation],
    text_units: list[TextUnit],
) -> tuple[list[str], str]:
    relevant_text_units = retrieve_text_units(
        query,
        text_units,
        top_k=3,
    )
    text_context = [text_unit.content for text_unit in relevant_text_units]
    context = get_context(text_context)
    response = llm.single_turn(ANSWER_PROMPT.format(question=query, context=context))
    if "I CANNOT ANSWER" not in response:
        return context, response

    # Get seed entity
    counter = Counter(entity for text_unit in text_units for entity in text_unit.entities)
    mean = sum(counter.values()) / len(counter.keys())
    seed_entities = set(k for k, v in counter.items() if v >= mean)
    print(len(seed_entities))
    
    # seed_entities.update(get_seed_entities(query, llm, entities, threshold["entity"]))
    # Retrieve subgraph
    subgraph = retrieve_subgraph(query, graph, seed_entities, relations, threshold["relation"])

    candidate_ids = set()
    id2text_unit = {text_unit.id: text_unit for text_unit in text_units}
    for node in subgraph.nodes():
        if entities[node].text_units is not None:
            candidate_ids.update(entities[node].text_units)

        if entities[node].alias is not None:
            for alia in entities[node].alias:
                if entities[alia].text_units is not None:
                    candidate_ids.update(entities[alia].text_units)

    candidate_text_units = [id2text_unit[id] for id in candidate_ids]
    relevant_text_units = retrieve_text_units(
        query,
        candidate_text_units,
        top_k=top_k,
    )
    text_context = [text_unit.content for text_unit in relevant_text_units]
    context = get_context(text_context, subgraph if required_graph else None)
    response = llm.single_turn(
        ANSWER_PROMPT.format(
            context=context,
            question=query)
    )
    return context, response.replace("\n\n", "\n")