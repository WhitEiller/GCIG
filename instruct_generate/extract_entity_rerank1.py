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
import itertools
from graphrag.llm import OpenAIModel
from graphrag.model import TextUnit
from graphrag.query import generate
from graphrag.query.loader import load_graph, load_parquet
from prompt import question_prompt2
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


text_unit_path = "/mnt/disk/yh24/test1/graphrag-purity/graph_hotpot/text_units/text_units-1-90.parquet"  # TODO: 文本块的路径
graph_dir = "/mnt/disk/yh24/test1/graphrag-purity/graph_hotpot/graph"  # TODO: 修改为graph的文件夹
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
used_pairs = set()  # 全局跟踪已使用的组合
start = 0
output_folder_question = "question_outputs"
os.makedirs(output_folder_question, exist_ok=True)  # 创建文件夹（如果不存在）
max_combinations = 100  # 限制生成的组合数量
combination_count = 0  # 计数器跟踪已生成的组合数量
for idx,ques in enumerate(tqdm(sorted_question[start:], desc="Processing questions")):
    # 检查是否已达到组合数量限制
    if combination_count >= max_combinations:
        print(f"已达到最大组合数量限制 {max_combinations}，停止生成")
        break
    
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
        top_k=3,
    )
    # 直接使用relevant_text_units_all生成两个相关文本
    selected_units = relevant_text_units_all
    
    if not selected_units:
        print(f"没有找到相关文本单元: {idx + start}")
        continue
    
    print(f"找到 {len(selected_units)} 个相关文本单元")
    
    # 如果只有一个文本单元，跳过
    if len(selected_units) < 2:
        print(f"文本单元数量少于2个，跳过: {idx + start}")
        continue
    
    # 生成所有可能的两两组合
    combinations = list(itertools.combinations(selected_units, 2))
    print(f"生成 {len(combinations)} 个两两组合")
    
    # 为每个组合生成问题文件
    for combo_idx, (unit1, unit2) in enumerate(combinations):
        # 检查是否已达到组合数量限制
        if combination_count >= max_combinations:
            print(f"已达到最大组合数量限制 {max_combinations}，停止生成")
            break
            
        # 检查两个文本单元的内容是否相同
        if unit1.content == unit2.content:
            print(f"跳过相同内容的组合: {unit1.id} + {unit2.id}")
            continue
        
        # 生成当前组合的唯一标识
        current_combination = frozenset([unit1.id, unit2.id])
        
        # 检查是否已经处理过这个组合
        if current_combination in used_pairs:
            print(f"跳过重复组合: {unit1.id} + {unit2.id}")
            continue
        
        # 记录当前组合
        used_pairs.add(current_combination)
        combination_count += 1  # 增加计数器
        
        required_graph = True
        text_context = [unit1.content, unit2.content]
        context,context_dict = get_context(text_context, subgraph if required_graph else None)
        qa = "Generate a reasonable question based on the entities and source and the following rules.\n"
        rule = "**Rules**\n" + question_prompt2 + "\n"
        sor = "**Sources**\n" + str(context_dict["Sources"]) + "\n"
        # print(len(sor))
        question_to = qa + rule + sor + "**Question**\n"
        output_filename = os.path.join(output_folder_question, f"question_{idx + start}_combo_{combo_idx}.txt")
        with open(output_filename, "w", encoding="utf-8") as q_file:
            q_file.write(question_to)
        
        print(f"已处理组合 {combo_idx + 1}/{len(combinations)} for question {idx + start} (总计: {combination_count}/{max_combinations})")
    
    # 检查是否已达到限制
    if combination_count >= max_combinations:
        print(f"已达到最大组合数量限制 {max_combinations}，停止处理")
        break
    
    print(f"问题 {idx + start} 完成处理，共生成 {len(combinations)} 个组合 (总计: {combination_count}/{max_combinations})")
