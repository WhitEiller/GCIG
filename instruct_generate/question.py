import os
import json
import asyncio
import aiofiles
from gpt import call_chatgpt
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

# 配置
INPUT_DIR = './question_outputs_model4'
MAX_CONCURRENT = 8  # 最大并发数
BATCH_SIZE = 100    # 每批保存的文件数
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

async def process_file(filename):
    """处理单个文件"""
    file_path = os.path.join(INPUT_DIR, filename)
    try:
        # 检查文件是否已经处理过（通过查找答案标记）
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            if "**Answer**" in content:
                return
        
        # 在线程池中运行GPT调用
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            executor,
            lambda: call_chatgpt(content)
        )
        if answer:
            # 追加答案到原文件
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                await f.write("\n\n**Answer**\n")
                # print(answer)
                await f.write(answer)
            
        # 添加小延迟避免API限制
        await asyncio.sleep(1)
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        # 记录错误到日志文件
        async with aiofiles.open('error_log.txt', 'a', encoding='utf-8') as f:
            await f.write(f"{filename}: {str(e)}\n")

async def process_batch(files):
    """处理一批文件"""
    tasks = [process_file(filename) for filename in files]
    await asyncio.gather(*tasks)

async def main():
    # 获取所有txt文件
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    total_files = len(files)
    print(f"Found {total_files} files to process")
    
    # 使用tqdm创建进度条
    with tqdm(total=total_files, desc="Processing files") as pbar:
        # 分批处理文件
        for i in range(0, total_files, BATCH_SIZE):
            batch = files[i:i + BATCH_SIZE]
            await process_batch(batch)
            pbar.update(len(batch))
            
            # 每批处理完后暂停一下，避免API限制
            if i + BATCH_SIZE < total_files:
                await asyncio.sleep(2)

if __name__ == "__main__":
    start_time = time.time()

    # 运行异步主函数
    asyncio.run(main())
    
    # 关闭线程池
    executor.shutdown()
    
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Results have been appended to the original files in {INPUT_DIR}") 
