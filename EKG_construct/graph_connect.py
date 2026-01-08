import json
import os
import time

import networkx as nx
import pandas as pd
from config import *
from tqdm import tqdm
from embedding import embedding
from graphrag.index import SentenceConnector

df = pd.read_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")

id2name = dict(zip(df["id"], df["name"]))

graph = nx.Graph()
for _, row in df.iterrows():
    graph.add_node(row["name"], id=row["id"])

connector = SentenceConnector(**ConnectorConfig.get_params())

# TODO: 获取文档列表[doc1, doc2, doc3]
# with open(corpus_path, "r", encoding="utf-8") as f:
#     corpus = json.load(f)

context = []
with open('//test1/graphrag-purity/input.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            context.append(json.loads(line)['input'])

print(f"总共加载了 {len(context)} 个文本块")
corpus = context[:15000]
print(len(corpus))
text_units = []
failed_count = 0
for text in tqdm(corpus, total=len(corpus)):
    try:
    # TODO: 获取每个文档的文本内容
    # text = text["body"]
        result = connector(graph, text, False)
        text_units.extend(result)
    except Exception as e:
        failed_count += 1
        print(f"处理文本时出错，已跳过 (错误 {failed_count}): {str(e)}")
        continue
    time.sleep(0.5)

print(f"处理完成，共跳过 {failed_count} 个有问题的文本")

df = pd.DataFrame(text_units)
df.to_parquet(os.path.join(text_unit_dir, f"text_units-{ConnectorConfig.expand}-{int(ConnectorConfig.overlap_threshold * 100)}.parquet"), engine="pyarrow")
print("connect done")
embedding(os.path.join(text_unit_dir, f"text_units-{ConnectorConfig.expand}-{int(ConnectorConfig.overlap_threshold * 100)}.parquet"), "content", "embedding")
print("embedding done")