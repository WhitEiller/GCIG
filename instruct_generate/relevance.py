import os
import asyncio
import aiofiles
import ast
import time
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from gpt import call_chatgpt
from prompt import relevant_prompt1


# 配置
INPUT_DIR = './question_outputs'
MAX_CONCURRENT = 4
BATCH_SIZE = 100
RELEVANCE_MARKER = '**Relevance**'

# 解析相关性结果的正则表达式
RELEVANCE_PATTERN = re.compile(r"(?:^|\n)(?:Answer:|\*\*Answer:\*\*)\s*(Yes|No)\b", re.IGNORECASE)


def extract_texts_between_markers(file_content: str):
    """从文件内容中抽取 **Sources** 与 **Question** 之间的两段文本。

    期望格式：在 **Sources** 和 **Question** 之间有一个 Python 风格的列表，包含两个字符串：
    ['Text A', 'Text B']
    """
    try:
        sources_index = file_content.find('**Sources**')
        if sources_index == -1:
            return None

        question_index = file_content.find('**Question**', sources_index + 1)
        if question_index == -1:
            return None

        block = file_content[sources_index + len('**Sources**'):question_index].strip()

        # 仅取方括号包裹的部分，使用 ast.literal_eval 解析为列表
        bracket_start = block.find('[')
        bracket_end = block.rfind(']')
        if bracket_start == -1 or bracket_end == -1 or bracket_end <= bracket_start:
            return None

        array_literal = block[bracket_start: bracket_end + 1]
        try:
            parsed = ast.literal_eval(array_literal)
        except Exception:
            return None

        if not isinstance(parsed, (list, tuple)) or len(parsed) < 2:
            return None

        text_a = str(parsed[0])
        text_b = str(parsed[1])
        return text_a, text_b
    except Exception:
        return None


def extract_relevance_result(text: str) -> str:
    """从GPT回答中提取相关性判断结果（Yes/No）。
    
    如果无法提取到明确的Yes/No结果，返回空字符串。
    """
    match = RELEVANCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return ""


async def process_file(filename: str):
    """处理单个 txt：抽取两段文本，调用评估提示词，写回结果。"""
    file_path = os.path.join(INPUT_DIR, filename)
    try:
        # 已处理过则跳过
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            if RELEVANCE_MARKER in content:
                return

        extracted = extract_texts_between_markers(content)
        if not extracted:
            # 记录无法解析的文件
            async with aiofiles.open('relevance_error_log.txt', 'a', encoding='utf-8') as ef:
                await ef.write(f"{filename}: cannot extract sources between markers\n")
            return

        text_a, text_b = extracted

        # 组装提示词
        prompt_text = relevant_prompt1.format(text_a=text_a, text_b=text_b)

        # 在线程池中调用 GPT
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            executor,
            lambda: call_chatgpt(prompt_text)
        )

        if answer:
            # 提取Yes/No结果
            relevance_result = extract_relevance_result(answer)
            
            # 写入完整回答和提取的结果
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                await f.write("\n\n" + RELEVANCE_MARKER + "\n")
                await f.write(answer)
                
                # 如果成功提取到了结果，添加一个明确的标记
                if relevance_result:
                    await f.write(f"\n\n**Relevance Result**: {relevance_result}")

        # 小延迟避免限流
        await asyncio.sleep(1)

    except Exception as e:
        async with aiofiles.open('relevance_error_log.txt', 'a', encoding='utf-8') as ef:
            await ef.write(f"{filename}: {str(e)}\n")


async def process_batch(files):
    tasks = [process_file(name) for name in files]
    await asyncio.gather(*tasks)


async def count_relevance_results():
    """统计已处理文件中的Yes/No结果数量"""
    yes_count = 0
    no_count = 0
    unknown_count = 0
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    
    for filename in files:
        try:
            file_path = os.path.join(INPUT_DIR, filename)
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                
                if RELEVANCE_MARKER not in content:
                    continue
                    
                # 查找"**Relevance Result**:"后的Yes/No结果
                result_match = re.search(r"\*\*Relevance Result\*\*:\s*(Yes|No)", content)
                if result_match:
                    result = result_match.group(1)
                    if result.lower() == "yes":
                        yes_count += 1
                    elif result.lower() == "no":
                        no_count += 1
                else:
                    unknown_count += 1
        except Exception:
            unknown_count += 1
            
    return yes_count, no_count, unknown_count


async def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    total_files = len(files)
    print(f"Found {total_files} files to process for relevance assessment")

    with tqdm(total=total_files, desc='Assessing relevance') as pbar:
        for i in range(0, total_files, BATCH_SIZE):
            batch = files[i:i + BATCH_SIZE]
            await process_batch(batch)
            pbar.update(len(batch))

            if i + BATCH_SIZE < total_files:
                await asyncio.sleep(2)
    
    # 统计处理结果
    yes_count, no_count, unknown_count = await count_relevance_results()
    processed_count = yes_count + no_count + unknown_count
    
    print("\nRelevance assessment results:")
    print(f"- Yes (texts are relevant): {yes_count}")
    print(f"- No (texts are not relevant): {no_count}")
    print(f"- Unknown/Unclear results: {unknown_count}")
    print(f"- Total processed: {processed_count} out of {total_files} files")


if __name__ == '__main__':
    start_time = time.time()

    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)
    try:
        asyncio.run(main())
    finally:
        executor.shutdown()

    duration = time.time() - start_time
    print(f"\nRelevance assessment completed in {duration:.2f} seconds")
    print(f"Results have been appended under {RELEVANCE_MARKER} in files within {INPUT_DIR}")
    print(f"Each file contains a Yes/No relevance judgment based on the expert content analyst prompt")

