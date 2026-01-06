import os
import pandas as pd
from config import *
from graphrag.utils.embedding import get_embedding
from tqdm import tqdm
import json
import numpy as np

def load_progress(progress_file):
    """加载进度文件"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"processed_ids": [], "last_index": -1}

def save_progress(progress_file, processed_ids, last_index):
    """保存进度"""
    progress_data = {
        "processed_ids": processed_ids,
        "last_index": last_index
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def embedding(file_path, column, embedding_column, batch_size=100):
    df = pd.read_parquet(file_path, engine="pyarrow")
    
    # 进度文件路径
    progress_file = file_path.replace('.parquet', '_progress.json')
    
    # 加载进度
    progress = load_progress(progress_file)
    processed_ids = set(progress["processed_ids"])
    start_index = progress["last_index"] + 1
    
    # 准备数据
    ids, content = [], []
    total_items = 0
    
    # 先统计需要处理的总数
    for i, item in enumerate(df[column]):
        if item is None or i in processed_ids:
            continue
        total_items += 1
    
    print(f"总共需要处理 {total_items} 条数据")
    
    # 如果embedding列不存在，创建它
    if embedding_column not in df.columns:
        df[embedding_column] = None
    else:
        # 将None值转换为NaN以便后续处理
        df[embedding_column] = df[embedding_column].apply(lambda x: None if x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0) else x)
    
    # 处理数据
    current_batch_ids = []
    current_batch_content = []
    processed_count = 0
    
    # 使用tqdm显示进度
    pbar = tqdm(total=total_items, initial=len(processed_ids), desc="Processing embeddings")
    
    for i, item in enumerate(df[column]):
        # 跳过已处理的数据
        if item is None or i in processed_ids:
            continue
            
        current_batch_ids.append(i)
        current_batch_content.append(item)
        
        # 当批次达到指定大小时，处理这一批数据
        if len(current_batch_content) >= batch_size:
            try:
                # 获取嵌入
                embeddings = get_embedding(current_batch_content)
                
                # 保存嵌入结果
                for j, embedding_result in enumerate(embeddings):
                    df.at[current_batch_ids[j], embedding_column] = embedding_result
                    processed_ids.add(current_batch_ids[j])
                
                # 更新进度
                processed_count += len(current_batch_ids)
                
                # 每批次更新进度条
                pbar.update(len(current_batch_ids))
                
                # 保存进度和数据
                save_progress(progress_file, list(processed_ids), max(current_batch_ids))
                df.to_parquet(file_path, engine="pyarrow")
                
                # 重置批次
                current_batch_ids = []
                current_batch_content = []
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                # 即使出错也保存已处理的数据
                save_progress(progress_file, list(processed_ids), max(current_batch_ids) if current_batch_ids else progress["last_index"])
                df.to_parquet(file_path, engine="pyarrow")
                continue
    
    # 处理剩余的数据
    if current_batch_content:
        try:
            embeddings = get_embedding(current_batch_content)
            
            # 保存嵌入结果
            for j, embedding_result in enumerate(embeddings):
                df.at[current_batch_ids[j], embedding_column] = embedding_result
                processed_ids.add(current_batch_ids[j])
            
            pbar.update(len(current_batch_content))
            
            # 最终保存
            save_progress(progress_file, list(processed_ids), max(current_batch_ids))
            df.to_parquet(file_path, engine="pyarrow")
            
        except Exception as e:
            print(f"处理最后批次时出错: {e}")
            save_progress(progress_file, list(processed_ids), max(current_batch_ids) if current_batch_ids else progress["last_index"])
            df.to_parquet(file_path, engine="pyarrow")
    
    pbar.close()
    
    # 清理进度文件（可选）
    # if os.path.exists(progress_file):
    #     os.remove(progress_file)
    
    print(f"处理完成！共处理 {len(processed_ids)} 条数据")

def main() -> None:
    embedding("/mnt/disk/yh24/test1/graphrag-purity/graph_PubMed/text_units/text_units-1-90.parquet", "content", "embedding", batch_size=100)

if __name__ == "__main__":
    main()