from dataclasses import dataclass
from typing import Any


@dataclass
class Entity:
    id: str
    name: str
    type: str | None = None
    description: str | None = None
    name_embedding: list[float] | None = None
    desc_embedding: list[float] | None = None
    text_units: list[str] | None = None
    alias: list[str] | None = None
    
    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        name_key: str = "name",
        description_key: str = "description",
        name_embedding_key: str = "name_embedding",
        desc_embedding_key: str = "desc_embedding",
        type_key: str = "type",
        text_units_key: str = "text_units",
        alias_key: str = "alias",
    ) -> "Entity":
        return Entity(
            id=d[id_key],
            name=d[name_key],
            description=d.get(description_key),
            type=d.get(type_key),
            name_embedding=d.get(name_embedding_key),
            desc_embedding=d.get(desc_embedding_key),
            text_units=d.get(text_units_key),
            alias=d.get(alias_key)
        )