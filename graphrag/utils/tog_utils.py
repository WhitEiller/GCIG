"""
ToG (Think-on-Graph) Utility Functions

This module contains utility functions extracted from the ToG algorithm,
adapted to work with the graphrag framework.

Reference:
    Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., Shum, H. Y., & Guo, J. (2024).
    Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph.
    In International Conference on Learning Representations (ICLR).
"""

import re
from typing import List, Tuple, Dict, Any

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def retrieve_top_docs(query: str, docs: List[str], model, width: int = 3) -> Tuple[List[str], List[float]]:
    """
    Retrieve the top-n most relevant documents for the given query using semantic similarity.

    Args:
        query: The input query string
        docs: List of documents to search from
        model: SentenceTransformer model for encoding
        width: Number of top documents to return

    Returns:
        Tuple of (top_docs, top_scores)
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers is required for retrieve_top_docs. "
                         "Install it with: pip install sentence-transformers")

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query: str, corpus: List[str], width: int = 3) -> Tuple[List[str], List[float]]:
    """
    Computes the BM25 similarity between a query and a corpus of documents.

    Args:
        query: Input query string
        corpus: List of documents (e.g., relations)
        width: Number of top documents to return

    Returns:
        Tuple of (top_relations, top_scores)
    """
    if not HAS_BM25:
        raise ImportError("rank-bm25 is required for compute_bm25_similarity. "
                         "Install it with: pip install rank-bm25")

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


def clean_relations(string: str, entity_id: str, head_relations: List[str]) -> Tuple[bool, Any]:
    """
    Parse LLM output to extract relations with scores.

    Args:
        string: LLM output string containing relations and scores
        entity_id: ID of the entity
        head_relations: List of head relations for the entity

    Returns:
        Tuple of (success, relations_list or error_message)
    """
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations = []

    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"

        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})

    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores: List[float]) -> bool:
    """Check if all scores are zero."""
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(
    topn_relations: List[str],
    topn_scores: List[float],
    entity_id: str,
    head_relations: List[str]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Clean relations obtained from BM25 or SentenceBERT ranking.

    Args:
        topn_relations: List of top relations
        topn_scores: Corresponding scores
        entity_id: ID of the entity
        head_relations: List of head relations

    Returns:
        Tuple of (success, relations_list)
    """
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)

    for i, relation in enumerate(topn_relations):
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})

    return True, relations


def all_unknown_entity(entity_candidates: List[str]) -> bool:
    """Check if all entity candidates are unknown."""
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates: List[str]) -> List[str]:
    """Remove unknown entities from candidate list."""
    if len(entity_candidates) == 1 and entity_candidates[0] == "UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(string: str, entity_candidates: List[str]) -> List[float]:
    """
    Extract numerical scores from LLM output.

    Args:
        string: LLM output containing scores
        entity_candidates: List of entity candidates

    Returns:
        List of scores (or uniform distribution if parsing fails)
    """
    scores = re.findall(r'\d+\.\d+', string)
    scores = [float(number) for number in scores]

    if len(scores) == len(entity_candidates):
        return scores
    else:
        # Uniform distribution fallback
        return [1.0 / len(entity_candidates)] * len(entity_candidates)
