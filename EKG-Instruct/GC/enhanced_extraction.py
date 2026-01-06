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
sys.path.append('/mnt/disk/yh24/test1/graphrag-purity')
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

splitter = TokenTextSplitter(**SplitterConfig.get_params())
llm = OpenAIModel(**LLMConfig.get_params())

# 优化extractor配置
extractor = GraphExtractor(
    llm=llm,
    entity_types=["PERSON", "ORGANIZATION", "PRODUCT", "LEGALTERM", "CONDITION"],
    max_gleanings=1
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
    elif len(chunk) > 4000:
        chunk = chunk[:4000] + "..."
    
    return chunk

def extract_entity_with_retry(entity_name, chunk_text, entity_types):
    """针对特定实体进行重新提取，确保获取类型和描述"""
    prompt = f"""
    From the following text, extract information about the entity "{entity_name}":
    
    Text: {chunk_text[:1000]}
    
    Please identify:
    1. The entity type (one of: {', '.join(entity_types)})
    2. A brief description of this entity based on the context
    
    Return the result in the following format:
    Entity: {entity_name}
    Type: [entity type]
    Description: [brief description]
    """
    
    try:
        response = llm.single_turn(prompt)
        
        # 解析响应
        lines = response.strip().split('\n')
        entity_type = None
        description = None
        
        for line in lines:
            if 'Type:' in line:
                entity_type = line.split('Type:')[1].strip()
                # 验证类型是否在允许的列表中
                if entity_type.upper() not in [t.upper() for t in entity_types]:
                    entity_type = "UNKNOWN"
            elif 'Description:' in line:
                description = line.split('Description:')[1].strip()
        
        return entity_type, description
    except Exception as e:
        logger.warning(f"Failed to re-extract entity {entity_name}: {e}")
        return None, None

def validate_and_fix_graph(graph, chunk_text):
    """验证并修复图中缺少类型或描述的实体"""
    fixed_nodes = []
    entity_types = ["PERSON", "ORGANIZATION", "PRODUCT", "LEGALTERM", "CONDITION"]
    
    for node, data in graph.nodes(data=True):
        needs_fix = False
        
        # 检查是否缺少type或description为空
        if 'type' not in data or not data.get('type'):
            needs_fix = True
            logger.info(f"Entity '{node}' missing type, attempting to fix...")
        
        if 'description' not in data or not data.get('description') or data['description'] == []:
            needs_fix = True
            logger.info(f"Entity '{node}' missing description, attempting to fix...")
        
        if needs_fix:
            # 重新提取这个实体的信息
            entity_type, description = extract_entity_with_retry(node, chunk_text, entity_types)
            
            if entity_type:
                data['type'] = entity_type
            else:
                data['type'] = "UNKNOWN"
            
            if description:
                data['description'] = [description]
            else:
                # 生成默认描述
                data['description'] = [f"Entity mentioned in the context"]
            
            fixed_nodes.append(node)
            logger.info(f"Fixed entity '{node}': type={data['type']}, description={data['description']}")
    
    return graph, fixed_nodes

def safe_extract(chunk, max_retries=3):
    """增强的实体提取函数，包含验证和修复机制"""
    
    # 预处理chunk
    processed_chunk = preprocess_chunk(chunk)
    if not processed_chunk:
        logger.warning(f"Chunk为空或过短，跳过处理")
        return nx.Graph()
    
    best_graph = None
    best_score = 0
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"开始提取实体 (尝试 {attempt + 1}/{max_retries})")
            
            # 尝试提取
            g = extractor(processed_chunk)
            
            # 验证并修复图
            g, fixed_nodes = validate_and_fix_graph(g, processed_chunk)
            
            # 计算图的质量分数
            score = 0
            total_nodes = len(g.nodes)
            
            if total_nodes > 0:
                # 计算有完整信息的节点比例
                complete_nodes = 0
                for node, data in g.nodes(data=True):
                    if data.get('type') and data.get('description') and data['description'] != []:
                        complete_nodes += 1
                
                score = complete_nodes / total_nodes
                
                logger.debug(f"提取结果: {total_nodes} 个节点, {len(g.edges)} 个边, 完整率: {score:.2%}")
                
                # 如果修复了节点，记录信息
                if fixed_nodes:
                    logger.info(f"修复了 {len(fixed_nodes)} 个不完整的实体")
            
            # 保存最佳结果
            if score > best_score:
                best_graph = g
                best_score = score
            
            # 如果完整率达到90%以上，直接返回
            if score >= 0.9:
                return g
            
        except Exception as e:
            logger.warning(f"提取失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
            
            if attempt == max_retries - 1 and best_graph is not None:
                # 返回最佳结果
                logger.info(f"返回最佳结果，完整率: {best_score:.2%}")
                return best_graph
            
            # 等待后重试
            time.sleep(1)
    
    # 返回最佳结果或空图
    if best_graph is not None:
        return best_graph
    else:
        logger.error(f"所有重试都失败，返回空图")
        return nx.Graph()

def merge(targ: nx.Graph, subgraph: nx.Graph):
    """线程安全的图合并函数"""
    try:
        for node, data in subgraph.nodes(data=True):
            if node in targ.nodes() and data.get("type") == targ.nodes[node].get("type"):
                # 安全地扩展描述列表
                if "description" in targ.nodes[node] and "description" in data:
                    # 合并描述，去重
                    existing_desc = set(targ.nodes[node]["description"])
                    new_desc = set(data["description"])
                    targ.nodes[node]["description"] = list(existing_desc | new_desc)
            else:
                targ.add_node(node, **data)
                
        for source, target, data in subgraph.edges(data=True):
            if targ.has_edge(source, target):
                # 安全地扩展关系列表
                if "relations" in targ.edges[source, target] and "relations" in data:
                    # 合并关系，去重
                    existing_rel = set(targ.edges[source, target]["relations"])
                    new_rel = set(data["relations"])
                    targ.edges[source, target]["relations"] = list(existing_rel | new_rel)
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
        
        # 保存前最后检查一次实体完整性
        incomplete_entities = []
        for node, data in graph.nodes(data=True):
            if not data.get('type') or not data.get('description'):
                incomplete_entities.append(node)
            
            ent2id[node] = id
            entities.append({
                "id": id,
                "name": node,
                "type": data.get("type", "UNKNOWN"),
                "description": data.get("description", [])
            })
            id += 1
        
        if incomplete_entities:
            logger.warning(f"保存时发现 {len(incomplete_entities)} 个不完整的实体: {incomplete_entities[:5]}")
        
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
        logger.info(f"保存成功: {len(entities)} 个实体, {len(relations)} 个关系")
    except Exception as e:
        logger.error(f"保存失败: {e}")

# 加载数据
context = []
with open('/mnt/disk/yh24/test1/graphrag-purity/input.json', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            context.append(json.loads(line)['input'])

print(f"总共加载了 {len(context)} 个文本块")

chunks = context
graph = nx.Graph()

# 尝试加载已有的图
if os.path.exists(os.path.join(graph_dir, "entities_enhanced.parquet")):
    try:
        df = pd.read_parquet(os.path.join(graph_dir, "entities_enhanced.parquet"), engine="pyarrow")
        id2name = dict(zip(df["id"], df["name"]))
        for index, record in df.iterrows():
            desc = record["description"]
            if isinstance(desc, str):
                desc = [desc]
            elif not isinstance(desc, list):
                desc = []
            graph.add_node(record["name"], type=record["type"], description=desc)
        
        df = pd.read_parquet(os.path.join(graph_dir, "relations_enhanced.parquet"), engine="pyarrow")
        for index, record in df.iterrows():
            desc = record["description"]
            if isinstance(desc, str):
                desc = [desc]
            elif not isinstance(desc, list):
                desc = []
            graph.add_edge(id2name[record["source"]], id2name[record["target"]], relations=desc)
        
        logger.info(f"加载已有图: {len(graph.nodes)} 个节点, {len(graph.edges)} 个边")
    except Exception as e:
        logger.warning(f"加载已有图失败: {e}")
        graph = nx.Graph()

# 全局统计变量
success_count = 0
error_count = 0
incomplete_count = 0
statistics_lock = threading.Lock()

def process_chunk(chunk_queue, graph, lock, progress_bar, error_queue):
    global success_count, error_count, incomplete_count
    
    while not chunk_queue.empty():
        i, chunk = chunk_queue.get()
        try:
            # 使用增强的提取函数
            g = safe_extract(chunk)
            
            # 统计不完整的实体
            local_incomplete = 0
            for node, data in g.nodes(data=True):
                if not data.get('type') or not data.get('description'):
                    local_incomplete += 1
            
            with lock:
                merge(graph, g)
                progress_bar.update(1)
                save(graph)
                
            with statistics_lock:
                success_count += 1
                if local_incomplete > 0:
                    incomplete_count += local_incomplete
                    logger.warning(f"Chunk {i}: 发现 {local_incomplete} 个不完整的实体")
                
        except Exception as e:
            logger.error(f"处理chunk {i}时发生意外错误: {e}")
            error_queue.put((i, e))
            with statistics_lock:
                error_count += 1
        finally:
            chunk_queue.task_done()

# 初始化处理队列
total_chunks = len(chunks)
start_index = 0
chunk_queue = Queue()
error_queue = Queue()
graph_lock = threading.Lock()

# 计算剩余要处理的chunks
remaining_chunks = total_chunks - start_index

# 填充队列
for i in range(start_index, total_chunks):
    chunk_queue.put((i, chunks[i]))

# 设置进度条
with tqdm(total=remaining_chunks, initial=0, desc=f"Processing chunks with enhanced extraction") as pbar:
    # 创建并启动工作线程
    num_threads = min(8, remaining_chunks)  # 适度的线程数
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=process_chunk,
            args=(chunk_queue, graph, graph_lock, pbar, error_queue)
        )
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # 监控和处理错误
    reported_errors = []
    while any(thread.is_alive() for thread in threads) or not error_queue.empty():
        if not error_queue.empty():
            i, error = error_queue.get()
            reported_errors.append((i, error))
            if len(reported_errors) <= 5:
                logger.error(f"Chunk {i} 处理错误: {str(error)[:100]}")
        time.sleep(0.1)
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 最终统计
    print(f"\n" + "="*60)
    print(f"增强提取处理完成统计:")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {error_count}")
    print(f"⚠️  不完整实体总数: {incomplete_count}")
    print(f"📊 成功率: {success_count/(success_count+error_count)*100:.1f}%")
    
    # 统计最终图的完整性
    complete_nodes = 0
    total_nodes = len(graph.nodes)
    for node, data in graph.nodes(data=True):
        if data.get('type') and data.get('type') != 'UNKNOWN' and data.get('description') and data['description'] != []:
            complete_nodes += 1
    
    print(f"🔗 最终图状态: {total_nodes} 个节点, {len(graph.edges)} 个边")
    print(f"📈 节点完整率: {complete_nodes}/{total_nodes} ({complete_nodes/total_nodes*100:.1f}%)")
    
    if len(reported_errors) > 5:
        print(f"⚠️  还有 {len(reported_errors) - 5} 个未显示的错误")
    print("="*60)

# 最终保存
with graph_lock:
    save(graph)
    print("\n✅ 增强提取完成，结果已保存到 entities_enhanced.parquet 和 relations_enhanced.parquet")