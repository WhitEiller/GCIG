import re
import uuid
from collections import defaultdict
from typing import Any

import networkx as nx
import spacy
import tiktoken


class SentenceConnector:
    def __init__(self,
                 expand: int = 1,
                 overlap_threshold: float = 0.8,
                 encoding_model: str = "cl100k_base") -> None:
        self.expand = expand
        self.overlap_threshold = overlap_threshold
        self._encoding_model = tiktoken.get_encoding(encoding_model)

    def _split(self, text: str) -> list[str]:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [sent.text for sent in doc.sents]

    def _get_pattern(self, word: str) -> str:
        return r"\s*\b" + re.escape(word) + r"(s?)\s*[!.,?;:']?"

    def _build_mapping(self,
                       entities: list[str],
                       sents: list[str],
                       ignore_case: bool = True) -> tuple[dict[str, Any], list[list[int]]]:
        sent2ents: list[list[str]] = [[] for _ in range(len(sents))]
        ent2sents: dict[str, set[int]] = defaultdict(set)
        # Mapping sentences containing entities.
        for entity in entities:
            # Search for sentences with entity appearing
            for id, sent in enumerate(sents):
                if ignore_case:
                    find = re.search(self._get_pattern(entity.lower()), sent.lower())
                else:
                    find = re.search(self._get_pattern(entity), sent)
                if find:
                    sent2ents[id].append(entity)
                    ent2sents[entity].add(id)

        # Mapping blank sentneces (devoid of entities).
        i = 0
        while i < len(sents):
            if not sent2ents[i]:
                # The closest sentence on the left containing entities
                j = i - 1
                # The closest sentence on the right containing entities
                while i < len(sents) and not sent2ents[i]:
                    i += 1
                # Associate blank sentences with entities that appear in the nearest sentences on both sides
                blank_sents = [id for id in range(j + 1, i)]
                if j >= 0:
                    for entity in sent2ents[j]:
                        ent2sents[entity].update(blank_sents)
                    for id in blank_sents:
                        sent2ents[id].extend(sent2ents[j])
                if i < len(sents):
                    for entity in sent2ents[i]:
                        ent2sents[entity].update(blank_sents)
                    for id in blank_sents:
                        sent2ents[id].extend(sent2ents[i])
            else:
                i += 1
        return ent2sents, sent2ents

    def _num_tokens(self, text: str) -> int:
        return len(self._encoding_model.encode(text))

    def _merge_sents(self, sent_ids: list[int], max_id: int) -> list[list[int]]:
        sent_ids.sort()
        pre = sent_ids[0]
        groups = [[i for i in range(max(0, pre - self.expand), pre + 1)]]
        for cur in sent_ids[1:]:
            # Sentences with a span of no more than 2 * expand can be merged.
            if cur <= pre + 2 * self.expand:
                groups[-1].extend(range(pre + 1, cur + 1))
            else:
                # Add `expand`(num) sentences on the right side to the current sentence group.
                groups[-1].extend(range(pre + 1, min(pre + self.expand, max_id) + 1))
                # Add sentences on the left side to the new sentence group.
                groups.append([i for i in range(cur - self.expand, cur + 1)])
            pre = cur
        groups[-1].extend(range(pre + 1, min(pre + self.expand, max_id) + 1))
        return groups

    def _reduce_text_slices(self, text_slices: list[tuple[int]]) -> list[tuple[int]]:
        text_slices.sort()
        merged = []
        cur = text_slices[0]
        for slice in text_slices[1:]:
            l1, r1 = cur
            l2, r2 = slice
            if r1 > l2:
                overlap_ratio = (r1 - l2 + 1) / (min(r1 - l1, r2 - l2) + 1)
                if overlap_ratio >= self.overlap_threshold:
                    cur = (l1, max(r1, r2))
                    continue
            merged.append(cur)
            cur = slice
        merged.append(cur)
        return merged

    def connect(self, graph: nx.Graph, text: str, ignore_case: bool = False) -> list[dict[str, Any]]:
        sents = self._split(text)
        ent2sents, sent2ents = self._build_mapping(graph.nodes(), sents, ignore_case)

        text_slices: set[tuple[int]] = set()
        for _, sent_ids in ent2sents.items():
            # Merge consecutive sentences.
            sent_groups = self._merge_sents(list(sent_ids), len(sents) - 1)
            for sent_group in sent_groups:
                l = sent_group[0]
                r = sent_group[-1]
                if (l, r) not in text_slices:
                    text_slices.add((l, r))
        if self.overlap_threshold == -1:
            text_slices = list(text_slices)
        else:
            text_slices = self._reduce_text_slices(list(text_slices))
        text_units = []
        for slice in text_slices:
            ent_list = set(entity for id in range(slice[0], slice[1] + 1) for entity in sent2ents[id])
            text_units.append({
                "id": str(uuid.uuid1()),
                "content": " ".join(sents[slice[0]: slice[1] + 1]),
                "entities": [graph.nodes[entity]["id"] for entity in ent_list]
            })
            for entity in ent_list:
                if "text_units" not in graph.nodes[entity]:
                    graph.nodes[entity]["text_units"] = []
                graph.nodes[entity]["text_units"].append(text_units[-1]["id"])
        return text_units

    def __call__(self, graph: nx.Graph, text: str, ignore_case: bool = False) -> list[dict[str, Any]]:
        return self.connect(graph, text, ignore_case)