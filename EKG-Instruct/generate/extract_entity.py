import sys
sys.path.append('/mnt/disk/yh24/test1/graphrag-purity')
from graphrag.llm import LLM
from graphrag.prompts.query.answer import ANSWER_PROMPT
from gpt import call_chatgpt
from graphrag.query.retrieval import *
from collections import Counter
import json
import sys
from datetime import datetime
import openai
import os
import random
from graphrag.llm import OpenAIModel
from graphrag.model import TextUnit
from graphrag.query import generate
from graphrag.query.loader import load_graph, load_parquet
from prompt import question_prompt
from tqdm import tqdm
def get_context(texts, graph=None):
    context_dict = {"Entities": [], "Relationships": [], "Sources": []}
    context = []
    
    if graph:
        context.append("="*5 + "Entities" + "="*5)
        for node in graph.nodes():
            entity_str = f"{graph.nodes[node]['name']}: {graph.nodes[node]['description']}"
            context.append(entity_str)
            context_dict["Entities"].append(entity_str)
        
        context.append("="*5 + "Relationships" + "="*5)
        for u, v in graph.edges():
            relationship_str = f"{u}, {v}: {graph.edges[u, v]['description']}"
            context.append(relationship_str)
            context_dict["Relationships"].append(relationship_str)
    
    context.append("="*5 + "Sources" + "="*5)
    for text in texts:
        context.append(text)
        context_dict["Sources"].append(text)
    
    return "\n".join(context), context_dict


text_unit_path = "/mnt/disk/yh24/test1/graphrag-purity/graph_privacy/text_units/text_units-1-65.parquet"  # TODO: 文本块的路径
graph_dir = "/mnt/disk/yh24/test1/graphrag-purity/graph_privacy/graph"  # TODO: 修改为graph的文件夹
graph, entities, relations = load_graph(graph_dir)
text_units = load_parquet(text_unit_path)
text_units = [TextUnit.from_dict(text_unit) for text_unit in text_units]


id2ent = {entity.id: entity for entity in entities}

for text_unit in text_units:
    for entity in text_unit.entities:
        # 检查 entity 是否在 id2ent 中
        if entity not in id2ent:
            print(f"Entity {entity} not found in id2ent, skipping...")
            continue  # 如果不存在，跳过当前循环迭代

        # 如果存在，继续处理
        if not id2ent[entity].text_units:
            id2ent[entity].text_units = []
        id2ent[entity].text_units.append(text_unit.id)
# 打开文件并逐行读取
question = [(entity.name,entity.alias) for entity in entities if entity.alias is not None]
sorted_question = sorted(question, key=lambda x: len(x[1]))  # 按 alias 的长度排序
sorted_question = sorted_question
print(len(sorted_question))
used_pairs = set()
start = 0
output_folder_question = "question_outputs"
os.makedirs(output_folder_question, exist_ok=True)  # 创建文件夹（如果不存在）
for idx,ques in enumerate(tqdm(sorted_question[start:], desc="Processing questions")):
    query = ques[0]
    relevant_text_units = retrieve_text_units(
        query,
        text_units,
        top_k=3,
    )
    seed_entities = set(entity_id for text_unit in relevant_text_units for entity_id in text_unit.entities)
    # print(len(seed_entities))
    # print(seed_entities)
    subgraph = retrieve_subgraph(query, graph, seed_entities, relations, 0.75)

    candidate_ids = set()
    id2text_unit = {text_unit.id: text_unit for text_unit in text_units}
    for node in subgraph.nodes():
        if entities[node].text_units is not None:
            candidate_ids.update(entities[node].text_units)

        if entities[node].alias is not None:
            for alia in entities[node].alias:
                if entities[alia].text_units is not None:
                    candidate_ids.update(entities[alia].text_units)
    # print(len(candidate_ids))
    candidate_text_units = [id2text_unit[id] for id in candidate_ids]
    # print(len(candidate_text_units))
    relevant_text_units_all  = retrieve_text_units(
        query,
        candidate_text_units,
        top_k=6,
    )
    all_combinations = [
        (a, b)
        for i, a in enumerate(relevant_text_units_all)
        for j, b in enumerate(relevant_text_units_all)
        if i < j and frozenset((a.id, b.id)) not in used_pairs
    ]

    if not all_combinations:
        print(f"Warning: No unused pairs available at idx {idx}")
        continue  # 或者跳过、或继续复用已有组合等

    # 确保选择的两个单元不相同
    while True:
        selected_pair = random.choice(all_combinations)
        if selected_pair[0] != selected_pair[1]:  # 确保两个单元不相同
            selected_units = [selected_pair[0], selected_pair[1]]
            break

    # 记录该组合
    used_pairs.add(frozenset((selected_pair[0].id, selected_pair[1].id)))
    required_graph = True
    text_context = [text_unit.content for text_unit in selected_units]
    context,context_dict = get_context(text_context, subgraph if required_graph else None)
    qa = "Generate a reasonable question based on the entities and source and the following rules.\n"
    rule = "**Rules**\n" + question_prompt + "\n"
    sor = "**Sources**\n" + str(context_dict["Sources"]) + "\n"
    # print(len(sor))
    question_to = qa + rule + sor + "**Question**\n"
    output_filename = os.path.join(output_folder_question, f"question_{idx + start}.txt")
    with open(output_filename, "w", encoding="utf-8") as q_file:
        q_file.write(question_to)

