import pandas as pd
import threading
from datetime import datetime
from config import *
from embedding import embedding
from graphrag.llm import OpenAIModel
from graphrag.prompts.index.summary import (ENTITY_DESC_SUMMARAY_PROMPT,
                                            RELATION_DESC_SUMMARY_PROMPT)

llm = OpenAIModel(**LLMConfig.get_params())
df_ent = pd.read_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")
id2name = dict(zip(df_ent["id"], df_ent["name"]))
df_rel = pd.read_parquet(os.path.join(graph_dir, "relations.parquet"), engine="pyarrow")

# 记录启动时间
start_time = datetime.now()
print(f"程序启动时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 线程锁，用于保护共享资源
lock = threading.Lock()

def process_entities_chunk(start_idx, end_idx, results):
    """处理指定范围内的实体描述"""
    chunk_descriptions = []
    for index in range(start_idx, end_idx):
        if index >= len(df_ent):
            break
        record = df_ent.iloc[index]
        desc_list = record["description"].tolist()
        if len(desc_list) == 0:
            chunk_descriptions.append((index, None))
        elif len(desc_list) == 1:
            chunk_descriptions.append((index, desc_list[0]))
        else:
            summary = llm.single_turn(
                ENTITY_DESC_SUMMARAY_PROMPT.format(entity=record["name"],text="\n\n".join(desc_list))
            )
            chunk_descriptions.append((index, summary))
    
    with lock:
        results.extend(chunk_descriptions)

def process_relations_chunk(start_idx, end_idx, results):
    """处理指定范围内的关系描述"""
    chunk_relations = []
    for index in range(start_idx, end_idx):
        if index >= len(df_rel):
            break
        record = df_rel.iloc[index]
        relation_list = record["description"].tolist()
        if len(relation_list) == 0:
            chunk_relations.append((index, None))
        elif len(relation_list) == 1:
            chunk_relations.append((index, relation_list[0]))
        else:
            summary = llm.single_turn(RELATION_DESC_SUMMARY_PROMPT.format(
                source=id2name[record["source"]],
                target=id2name[record["target"]],
                text="\n\n".join(relation_list)
            ))
            chunk_relations.append((index, summary))
    
    with lock:
        results.extend(chunk_relations)

def process_entities_parallel():
    """使用8个线程并行处理实体"""
    num_threads = 2
    chunk_size = len(df_ent) // num_threads + 1
    threads = []
    results = []
    
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_ent))
        if start_idx < len(df_ent):
            thread = threading.Thread(target=process_entities_chunk, args=(start_idx, end_idx, results))
            threads.append(thread)
            thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 更新DataFrame
    descriptions = [None] * len(df_ent)
    for idx, desc in results:
        descriptions[idx] = desc
    
    df_ent["description"] = descriptions
    df_ent.to_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")
    print("summary_entities done")

def process_relations_parallel():
    """使用8个线程并行处理关系"""
    num_threads = 2
    chunk_size = len(df_rel) // num_threads + 1
    threads = []
    results = []
    
    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df_rel))
        if start_idx < len(df_rel):
            thread = threading.Thread(target=process_relations_chunk, args=(start_idx, end_idx, results))
            threads.append(thread)
            thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 更新DataFrame
    relations = [None] * len(df_rel)
    for idx, rel in results:
        relations[idx] = rel
    
    df_rel["description"] = relations
    df_rel.to_parquet(os.path.join(graph_dir, "relations.parquet"), engine="pyarrow")
    print("summary_relations done")

# 开始并行处理
print("开始并行处理entities和relations...")

# 创建主处理线程
entity_thread = threading.Thread(target=process_entities_parallel)
relation_thread = threading.Thread(target=process_relations_parallel)

# 启动线程
entity_thread.start()
relation_thread.start()

# 等待两个主处理线程完成
entity_thread.join()
relation_thread.join()

print("所有摘要处理完成，开始embedding...")

# 处理完成后进行embedding
embedding(os.path.join(graph_dir, "entities.parquet"), "description", "embedding")
print("embedding_entities done")
embedding(os.path.join(graph_dir, "relations.parquet"), "description", "embedding")
print("embedding_relations done")

# 记录完成时间并计算耗时
end_time = datetime.now()
duration = end_time - start_time

print("所有处理完成！")
print(f"程序完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总耗时: {duration}")
print(f"总耗时（秒）: {duration.total_seconds():.2f}秒")