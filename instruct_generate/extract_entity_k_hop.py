import sys
sys.path.append('//test1/graphrag-purity')
from graphrag.llm import LLM
from graphrag.prompts.query.answer import ANSWER_PROMPT
from gpt import call_chatgpt
from graphrag.query.retrieval import *
from collections import Counter, deque
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
from tqdm import tqdm
import networkx as nx

def get_n_hop_subgraph(graph, start_node_id, n_hops):
    """
    生成从指定节点开始的n跳子图
    
    Args:
        graph: NetworkX图对象
        start_node_id: 起始节点ID
        n_hops: 跳数
    
    Returns:
        subgraph: 子图对象
        subgraph_nodes: 子图中的节点列表
    """
    if start_node_id not in graph.nodes():
        print(f"警告：节点 {start_node_id} 不在图中")
        return None, []
    
    # 使用BFS查找n跳内的所有节点
    visited_nodes = set()
    queue = deque([(start_node_id, 0)])  # (节点ID, 当前跳数)
    visited_nodes.add(start_node_id)
    
    while queue:
        current_node, current_hop = queue.popleft()
        
        if current_hop < n_hops:
            # 添加邻居节点
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, current_hop + 1))
    
    # 创建子图
    subgraph_nodes = list(visited_nodes)
    subgraph = graph.subgraph(subgraph_nodes).copy()
    
    print(f"从节点 {start_node_id} 开始的 {n_hops} 跳子图包含 {len(subgraph_nodes)} 个节点和 {len(subgraph.edges())} 条边")
    
    return subgraph, subgraph_nodes

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


text_unit_path = "//test1/graphrag-purity/graph_hotpot/text_units/text_units-1-90.parquet"  # TODO: 文本块的路径
graph_dir = "//test1/graphrag-purity/graph_hotpot/graph"  # TODO: 修改为graph的文件夹
graph, entities, relations = load_graph(graph_dir)
text_units = load_parquet(text_unit_path)
text_units = [TextUnit.from_dict(text_unit) for text_unit in text_units]
print("原始图信息:")
print(f"总节点数: {len(graph.nodes())}")
print(f"总边数: {len(graph.edges())}")
print(f"总实体数: {len(entities)}")


# 设置跳数
n_hops = 1  # 您可以修改这个值来改变跳数

# 获取起始实体的ID
start_entity = entities[642]
start_node_id = start_entity.id

print(f"\n从实体 '{start_entity.name}' (ID: {start_node_id}) 开始生成 {n_hops} 跳子图...")

# 生成子图
subgraph, subgraph_nodes = get_n_hop_subgraph(graph, start_node_id, n_hops)

if subgraph is not None:
    
    relevant_text_units = []
    subgraph_entity_ids = set(subgraph_nodes)
    
    print(f"\n查找与子图实体相关的文本单元...")
    print(f"子图包含实体ID: {subgraph_entity_ids}")
    
    selected_text_units_info = []  # 存储被选中的文本单元信息
    
    for text_unit in text_units:
        # 检查文本单元是否包含子图中的任何实体
        if hasattr(text_unit, 'entities') and text_unit.entities is not None:
            # 将numpy数组转换为Python列表再转为集合
            text_unit_entity_list = text_unit.entities.tolist() if hasattr(text_unit.entities, 'tolist') else list(text_unit.entities)
            text_unit_entities = set(text_unit_entity_list)
            # 如果有交集，说明这个文本单元与子图中的实体相关
            overlap = text_unit_entities.intersection(subgraph_entity_ids)
            if overlap:
                relevant_text_units.append(text_unit.content)
                selected_text_units_info.append({
                    'id': text_unit.id,
                    'entities': text_unit_entity_list,
                    'overlap': list(overlap)
                })
    
    print(f"\n实际被选中的前3个文本单元信息:")
    for i, unit_info in enumerate(selected_text_units_info[:3]):
        print(f"选中文本单元 {i}: {unit_info['id']}, 包含实体: {unit_info['entities'][:10]}..., 与子图重叠实体: {unit_info['overlap']}")
    
    print(f"总共找到 {len(relevant_text_units)} 个相关文本单元")
    
    # 使用子图和相关文本单元生成上下文
    subgraph_context, context_dict = get_context(relevant_text_units, subgraph)
    
    # print(context_dict)
    print(f"\n生成的子图上下文长度: {len(subgraph_context)} 字符")