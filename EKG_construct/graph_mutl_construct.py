import os
import sys
import json
from config import *
import threading
from queue import Queue
from tqdm import tqdm
import time
import logging
import re
sys.path.append('//test1/graphrag-purity')
import networkx as nx
import pandas as pd
from tqdm import tqdm

from graphrag.index import GraphExtractor, TokenTextSplitter
from graphrag.llm import OpenAIModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用HTTP请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# text = open(text_path, "r", encoding="utf-8").read()

splitter = TokenTextSplitter(**SplitterConfig.get_params())

llm = OpenAIModel(**LLMConfig.get_params())

# 优化extractor配置：减少max_gleanings以提高稳定性
extractor = GraphExtractor(
    llm=llm,
    entity_types=["PERSON", "ORGANIZATION", "PRODUCT", "LEGALTERM", "CONDITION"],
    max_gleanings=1  # 减少为1，避免多轮对话导致的格式问题
)

def preprocess_chunk(chunk):
    """预处理chunk文本，清理可能导致问题的字符"""
    if not chunk or not isinstance(chunk, str):
        return ""
    
    # 移除控制字符
    chunk = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', chunk)
    
    # 清理多余的空白字符
    chunk = re.sub(r'\s+', ' ', chunk).strip()
    
    # 如果chunk太短或太长，进行处理
    if len(chunk) < 10:
        return ""
    elif len(chunk) > 4000:  # 截断过长的文本
        chunk = chunk[:4000] + "..."
    
    return chunk

def safe_extract(chunk, max_retries=2):  # 减少重试次数
    """安全的实体提取函数，包含重试机制"""
    
    # 预处理chunk
    processed_chunk = preprocess_chunk(chunk)
    if not processed_chunk:
        logger.warning(f"Chunk为空或过短，跳过处理")
        return nx.Graph()
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"开始提取实体 (尝试 {attempt + 1}/{max_retries})")
            
            # 尝试提取
            g = extractor(processed_chunk)
            
            # 验证结果
            if g and (len(g.nodes) > 0 or len(g.edges) > 0):
                logger.debug(f"成功提取: {len(g.nodes)} 个节点, {len(g.edges)} 个边")
            else:
                logger.debug("提取到空图")
            
            return g
            
        except Exception as e:
            logger.warning(f"提取失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
            
            if attempt == max_retries - 1:
                # 最后一次尝试失败，返回空图
                logger.error(f"所有重试都失败，返回空图")
                return nx.Graph()
            
            # 等待后重试
            time.sleep(0.5)  # 减少等待时间
    
    return nx.Graph()

def merge(targ: nx.Graph, subgraph: nx.Graph):
    """线程安全的图合并函数"""
    try:
        for node, data in subgraph.nodes(data=True):
            if node in targ.nodes() and data.get("type") == targ.nodes[node].get("type"):
                # 安全地扩展描述列表
                if "description" in targ.nodes[node] and "description" in data:
                    targ.nodes[node]["description"].extend(data["description"])
            else:
                targ.add_node(node, **data)
                
        for source, target, data in subgraph.edges(data=True):
            if targ.has_edge(source, target):
                # 安全地扩展关系列表
                if "relations" in targ.edges[source, target] and "relations" in data:
                    targ.edges[source, target]["relations"].extend(data["relations"])
            else:
                targ.add_edge(source, target, **data)
    except Exception as e:
        logger.error(f"图合并失败: {e}")

def save(graph):
    """保存图数据到文件"""
    try:
        ent2id = {}
        entities = []
        id = 0
        for node, data in graph.nodes(data=True):
            ent2id[node] = id
            entities.append({
                "id": id,
                "name": node,
                **data
            })
            id += 1
        relations = []
        id = 0
        for u, v, data in graph.edges(data=True):
            relations.append({
                "id": id,
                "source": ent2id[u],
                "target": ent2id[v],
                "description": graph.edges[u, v].get("relations", [])
            })
            id += 1
        
        # 确保目录存在
        os.makedirs(graph_dir, exist_ok=True)
        
        pd.DataFrame(entities).to_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")
        pd.DataFrame(relations).to_parquet(os.path.join(graph_dir, "relations.parquet"), engine="pyarrow")
        logger.debug(f"保存成功: {len(entities)} 个实体, {len(relations)} 个关系")
    except Exception as e:
        logger.error(f"保存失败: {e}")


context = []
with open('//test1/graphrag-purity/input.json', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            context.append(json.loads(line)['input'])
print(len(context))
chunks = context[:15000]
graph = nx.Graph()
if os.path.exists(os.path.join(graph_dir, "entities.parquet")):
    df = pd.read_parquet(os.path.join(graph_dir, "entities.parquet"), engine="pyarrow")
    id2name = dict(zip(df["id"], df["name"]))
    for index, record in df.iterrows():
        graph.add_node(record["name"], type=record["type"], description=record["description"].tolist())

    df = pd.read_parquet(os.path.join(graph_dir, "relations.parquet"), engine="pyarrow")
    for index, record in df.iterrows():
        graph.add_edge(id2name[record["source"]], id2name[record["target"]], relations=record["description"].tolist())

# 全局统计变量
success_count = 0
error_count = 0
statistics_lock = threading.Lock()

def process_chunk(chunk_queue, graph, lock, progress_bar, error_queue):
    global success_count, error_count
    
    while not chunk_queue.empty():
        i, chunk = chunk_queue.get()
        try:
            # 使用安全的提取函数
            g = safe_extract(chunk)
            
            with lock:
                merge(graph, g)
                progress_bar.update(1)
                save(graph)
                
            with statistics_lock:
                success_count += 1
                
        except Exception as e:
            logger.error(f"处理chunk {i}时发生意外错误: {e}")
            error_queue.put((i, e))
            with statistics_lock:
                error_count += 1
        finally:
            chunk_queue.task_done()

# Initialize
total_chunks = len(chunks)
start_index = 0  # Starting from chunk 32
chunk_queue = Queue()
error_queue = Queue()
graph_lock = threading.Lock()

# Calculate remaining chunks to process
remaining_chunks = total_chunks - start_index

# Fill the queue with chunks starting from index 32
for i in range(start_index, total_chunks):
    chunk_queue.put((i, chunks[i]))

# Set up progress bar
with tqdm(total=remaining_chunks, initial=0, desc=f"Processing chunks (starting from {start_index})") as pbar:
    # Create and start worker threads - 减少线程数以提高稳定性
    num_threads = min(15, remaining_chunks)  # 减少到4个线程
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=process_chunk,
            args=(chunk_queue, graph, graph_lock, pbar, error_queue)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Monitor and handle errors
    reported_errors = []
    while any(thread.is_alive() for thread in threads) or not error_queue.empty():
        if not error_queue.empty():
            i, error = error_queue.get()
            reported_errors.append((i, error))
            if len(reported_errors) <= 5:  # 只报告前5个错误，避免刷屏
                logger.error(f"Chunk {i} 处理错误: {str(error)[:100]}")
        time.sleep(0.1)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # 最终统计
    print(f"\n" + "="*50)
    print(f"处理完成统计:")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {error_count}")
    print(f"📊 成功率: {success_count/(success_count+error_count)*100:.1f}%")
    print(f"🔗 当前图状态: {len(graph.nodes)} 个节点, {len(graph.edges)} 个边")
    if len(reported_errors) > 5:
        print(f"⚠️  还有 {len(reported_errors) - 5} 个未显示的错误")
    print("="*50)

# Final save
with graph_lock:
    save(graph)
