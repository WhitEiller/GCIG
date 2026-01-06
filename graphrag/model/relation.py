from dataclasses import dataclass
from typing import Any


@dataclass
class Relation:
    source: str
    target: str
    description: str
    id: str | None = None
    embedding: list[float] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        source_key: str = "source",
        target_key: str = "target",
        description_key: str = "description",
        embedding_key: str = "embedding",
    ) -> "Relation":
        return Relation(
            source=d[source_key],
            target=d[target_key],
            id=d.get(id_key),
            description=d[description_key],
            embedding=d.get(embedding_key),
        )