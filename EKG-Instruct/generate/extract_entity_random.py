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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from prompt import relevant_prompt
from gpt import call_chatgpt

def get_relevant_prompt(text_a, text_b):
    prompt = relevant_prompt.format(text_a=text_a, text_b=text_b)
    return call_chatgpt(prompt)

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

def find_most_similar_text_unit(target_unit, text_units, embeddings_matrix, used_pairs):
    """
    找到与目标文本单元最相似的文本单元
    使用预计算的embedding向量
    """
    # 获取目标单元的embedding向量
    target_embedding = target_unit.embedding.reshape(1, -1)
    
    # 计算与所有其他文本单元的余弦相似度
    similarities = cosine_similarity(target_embedding, embeddings_matrix).flatten()
    
    # 获取按相似度排序的索引
    similar_indices = np.argsort(similarities)[::-1]
    
    # 寻找最相似且未使用过的文本单元
    for idx in similar_indices:
        candidate_unit = text_units[idx]
        
        # 跳过自己
        if candidate_unit.id == target_unit.id:
            continue
            
        # 跳过内容相同的
        if candidate_unit.content == target_unit.content:
            continue
            
        # 检查是否已经使用过这个组合（基于文本内容）
        current_combination = frozenset([target_unit.content, candidate_unit.content])
        if current_combination not in used_pairs:
            return candidate_unit, similarities[idx]
    
    return None, 0

# 加载数据
text_unit_path = "/mnt/disk/yh24/test1/graphrag-purity/graph_PubMed/text_units/text_units-1-65.parquet"
graph_dir = "/mnt/disk/yh24/test1/graphrag-purity/graph_PubMed/graph"
graph, entities, relations = load_graph(graph_dir)
text_units = load_parquet(text_unit_path)
text_units = [TextUnit.from_dict(text_unit) for text_unit in text_units]

print(f"加载了 {len(text_units)} 个文本单元")
print(f"示例文本单元: {text_units[0]}")

# 预处理embedding矩阵用于相似度计算
print("正在提取embedding向量...")
embeddings_matrix = np.array([unit.embedding for unit in text_units])
print(f"Embedding矩阵形状: {embeddings_matrix.shape}")
print("Embedding向量提取完成")

# 创建输出文件夹
output_folder_question = "question_outputs"
os.makedirs(output_folder_question, exist_ok=True)

# 全局跟踪已使用的组合
used_pairs = set()
successful_pairs = 0
max_iterations = 15000

print(f"开始生成 {max_iterations} 个文本单元对...")

for iteration in tqdm(range(max_iterations), desc="生成文本单元对"):
    # 随机选择一个文本单元作为起始点
    random_unit = random.choice(text_units)
    
    # 找到与之最相似的文本单元
    similar_unit, similarity_score = find_most_similar_text_unit(
        random_unit, text_units, embeddings_matrix, used_pairs
    )
    
    if similar_unit is None:
        print(f"迭代 {iteration}: 未找到合适的相似文本单元")
        continue
    
    # 记录当前组合（基于文本内容）
    current_combination = frozenset([random_unit.content, similar_unit.content])
    used_pairs.add(current_combination)
    
    # 生成上下文
    required_graph = True
    text_context = [random_unit.content, similar_unit.content]
    
    # 获取相关的子图（如果需要）
    subgraph = None
    if required_graph:
        # 简化的子图获取逻辑，基于文本单元的实体
        seed_entities = set()
        if hasattr(random_unit, 'entities') and random_unit.entities is not None and len(random_unit.entities) > 0:
            seed_entities.update(random_unit.entities)
        if hasattr(similar_unit, 'entities') and similar_unit.entities is not None and len(similar_unit.entities) > 0:
            seed_entities.update(similar_unit.entities)
        
        if seed_entities:
            # 创建包含相关实体的子图
            subgraph_nodes = []
            for entity_id in seed_entities:
                if entity_id in graph.nodes():
                    subgraph_nodes.append(entity_id)
            
            if subgraph_nodes:
                subgraph = graph.subgraph(subgraph_nodes)
    
    context, context_dict = get_context(text_context, subgraph if required_graph else None)
    
    # 生成问题提示
    qa = "Generate a reasonable question based on the entities and source and the following rules.\n"
    rule = "**Rules**\n" + question_prompt2 + "\n"
    sor = "**Sources**\n" + str(context_dict["Sources"]) + "\n"
    question_to = qa + rule + sor + "**Question**\n"
    
    # 保存到文件
    output_filename = os.path.join(output_folder_question, f"question_pair_{iteration}_similarity_{similarity_score:.3f}.txt")
    with open(output_filename, "w", encoding="utf-8") as q_file:
        q_file.write(question_to)
    successful_pairs += 1
    
    if iteration % 50 == 0:
        print(f"已完成 {iteration} 次迭代，成功生成 {successful_pairs} 个文本单元对")

print(f"\n总共完成 {max_iterations} 次迭代")
print(f"成功生成 {successful_pairs} 个唯一的文本单元对")
print(f"使用的组合总数: {len(used_pairs)}")
print(f"输出文件保存在: {output_folder_question}")