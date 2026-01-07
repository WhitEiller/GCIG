import os
import sys

import pandas as pd
from config import *

from graphrag.index import Aligner, llm_align, similarity_align, type_align
from graphrag.llm import OpenAIModel
from graphrag.model import Entity

llm = OpenAIModel(**LLMConfig.get_params())

aligner = Aligner()
aligner.add(type_align)
aligner.add(similarity_align, **SimilarityAlignConfig.get_params())
aligner.add(llm_align, llm=llm)

df = pd.read_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")
entities = [
    Entity(
        id=record["id"],
        name=record["name"],
        description=record["description"],
        type=record["type"],
        desc_embedding=record["embedding"].tolist(),
    )
    for _, record in df.iterrows() if record["type"] != None and record["embedding"] is not None
]
aligner.run(entities)

id = 0
alias = []
for i in df["id"]:
    if id < len(entities) and entities[id].id == i:
        alias.append(entities[id].alias)
        id += 1
    else:
        alias.append(None)

df["alias"] = alias
df.to_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")