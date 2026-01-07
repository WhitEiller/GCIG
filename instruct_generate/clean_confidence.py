import os
import re
from typing import List, Tuple, Dict, Optional


# 基础配置
INPUT_DIR = './question_outputs'
OUTPUT_DIR = './question_outputs_filtered'
CONFIDENCE_THRESHOLD = 0   # 过滤阈值：设为0表示不过滤置信度
KEEP_TOP_K = 3000          # 保留条数
RELEVANCE_MARKER = '**Relevance**'  # 从该标记起截断内容
RELEVANCE_RESULT_MARKER = '**Relevance Result**'  # 相关性结果标记


CONFIDENCE_PATTERN = re.compile(r"Confidence\s*(?:\([^)]*\))?\s*:\s*(\d+)\s*%", re.IGNORECASE)
RELEVANCE_RESULT_PATTERN = re.compile(r"\*\*Relevance Result\*\*:\s*(Yes|No)", re.IGNORECASE)


def extract_confidences(text: str) -> List[int]:
    """提取文本中所有 Confidence 的百分数数值（整数）。

    匹配示例：
    - Confidence (A): 57%
    - Confidence (B): 57%
    - Confidence (A+B): 100%
    - Confidence: 75%
    """
    return [int(x) for x in CONFIDENCE_PATTERN.findall(text)]


def extract_relevance_result(text: str) -> Optional[str]:
    """从文本中提取相关性判断结果（Yes/No）。
    
    如果找到明确的Yes/No结果，返回"Yes"或"No"；
    如果未找到，返回None。
    """
    match = RELEVANCE_RESULT_PATTERN.search(text)
    if match:
        return match.group(1)
    return None


def is_valid_by_threshold(confidences: List[int], threshold: int) -> bool:
    """若存在任何一项 < threshold，则判为无效。"""
    if not confidences:
        return False
    return all(value >= threshold for value in confidences)


def score_by_sum(confidences: List[int]) -> int:
    """按题意：按所有 Confidence 的总分排序。"""
    return sum(confidences)


def list_txt_files(directory: str) -> List[str]:
    return [f for f in os.listdir(directory) if f.endswith('.txt')]


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def strip_from_marker(text: str, marker: str) -> str:
    """从 marker 开始（含 marker）到文本末尾全部删除。若未找到 marker，原样返回。"""
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[:idx].rstrip() + "\n"


def filter_and_copy(
    input_dir: str,
    output_dir: str,
    threshold: int,
    keep_top_k: int,
    relevance_filter: Optional[str] = None,  # 可选参数：'yes', 'no', None (不过滤)
):
    files = list_txt_files(input_dir)

    # 结构扩展为：(sum_confidence, max_confidence, relevance_result, filename)
    # relevance_result 可以是 'Yes', 'No', 或 None
    scored_items: List[Tuple[int, int, Optional[str], str]] = []

    # 相关性统计与收集 yes 项
    relevance_stats = {'yes': 0, 'no': 0, 'unknown': 0}
    yes_filenames: List[str] = []

    for filename in files:
        full_path = os.path.join(input_dir, filename)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        # 提取相关性结果
        relevance_result = extract_relevance_result(content)
        if relevance_result:
            if relevance_result.lower() == 'yes':
                relevance_stats['yes'] += 1
                yes_filenames.append(filename)
            elif relevance_result.lower() == 'no':
                relevance_stats['no'] += 1
        else:
            relevance_stats['unknown'] += 1

        # 如果指定了相关性过滤条件，则筛选
        if relevance_filter and relevance_result:
            if relevance_result.lower() != relevance_filter.lower():
                continue

        # 提取并检查置信度
        confidences = extract_confidences(content)
        if not is_valid_by_threshold(confidences, threshold):
            continue

        total = score_by_sum(confidences)
        max_c = max(confidences) if confidences else 0
        scored_items.append((total, max_c, relevance_result, filename))

    # 按总分降序，其次按最大值降序，最后按文件名升序
    scored_items.sort(key=lambda x: (-x[0], -x[1], x[3]))

    # 保留前 K 个
    kept = scored_items[:keep_top_k]

    ensure_output_dir(output_dir)

    # 写出清理后的文件（去除 **Relevance** 及其后的内容）
    for _, __, ___, fname in kept:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        try:
            with open(src, 'r', encoding='utf-8') as f:
                original = f.read()
            cleaned = strip_from_marker(original, RELEVANCE_MARKER)
            with open(dst, 'w', encoding='utf-8') as wf:
                wf.write(cleaned)
        except Exception:
            # 忽略单个失败
            pass

    # 将 input 中所有 Relevance=Yes 的文件名写入目标文件
    # try:
    #     yes_items_path = os.path.join(output_dir, 'yes_items.txt')
    #     with open(yes_items_path, 'w', encoding='utf-8') as yf_list:
    #         for name in yes_filenames:
    #             yf_list.write(f"{name}\n")
    # except Exception:
    #     pass

    # 复制所有 Yes 文件到目标目录（同样做去除 **Relevance** 及其后内容的清理）
    copied_yes = 0
    for fname in yes_filenames:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        try:
            with open(src, 'r', encoding='utf-8') as f:
                original = f.read()
            cleaned = strip_from_marker(original, RELEVANCE_MARKER)
            with open(dst, 'w', encoding='utf-8') as wf:
                wf.write(cleaned)
            copied_yes += 1
        except Exception:
            # 忽略单个失败
            pass

    # 控制台输出摘要
    total_files = len(files)
    after_threshold = len(scored_items)
    kept_count = len(kept)
    
    print(f"Total txt files: {total_files}")
    print(f"Relevance statistics:")
    print(f"- Yes: {relevance_stats['yes']}")
    print(f"- No: {relevance_stats['no']}")
    print(f"- Unknown: {relevance_stats['unknown']}")
    
    # 将整体的 Yes 统计写入输出目录中的文件，便于后续查阅
    # try:
    #     yes_count_path = os.path.join(output_dir, 'yes_count.txt')
    #     with open(yes_count_path, 'w', encoding='utf-8') as yf:
    #         yf.write(f"- Yes: {relevance_stats['yes']}\n")
    # except Exception:
    #     pass
    
    if relevance_filter:
        print(f"Applied relevance filter: {relevance_filter}")
    
    print(f">= {threshold}% on all confidences: {after_threshold}")
    print(f"Kept top {keep_top_k}: {kept_count}")
    
    if kept:
        best_total = kept[0][0]
        worst_total = kept[-1][0]
        print(f"Top-kept total confidence range: {worst_total} ~ {best_total}")
        
        # 统计保留文件中的相关性分布
        kept_yes = sum(1 for _, __, rel, ___ in kept if rel and rel.lower() == 'yes')
        kept_no = sum(1 for _, __, rel, ___ in kept if rel and rel.lower() == 'no')
        kept_unknown = sum(1 for _, __, rel, ___ in kept if not rel)
        print(f"Relevance distribution in kept files:")
        print(f"- Yes: {kept_yes}")
        print(f"- No: {kept_no}")
        print(f"- Unknown: {kept_unknown}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='过滤和清理数据集')
    parser.add_argument('--relevance', choices=['yes', 'no'], help='根据相关性结果过滤（yes 或 no）')
    parser.add_argument('--threshold', type=int, default=CONFIDENCE_THRESHOLD, help=f'置信度阈值（默认 {CONFIDENCE_THRESHOLD}%）')
    parser.add_argument('--keep', type=int, default=KEEP_TOP_K, help=f'保留前K个文件（默认 {KEEP_TOP_K}）')
    parser.add_argument('--input', default=INPUT_DIR, help=f'输入目录（默认 {INPUT_DIR}）')
    parser.add_argument('--output', default=OUTPUT_DIR, help=f'输出目录（默认 {OUTPUT_DIR}）')
    
    args = parser.parse_args()
    
    # 默认提取 "yes" 相关性的项目
    relevance_filter = args.relevance if args.relevance else 'yes'
    
    filter_and_copy(
        input_dir=args.input,
        output_dir=args.output,
        threshold=args.threshold,
        keep_top_k=args.keep,
        relevance_filter=relevance_filter,
    )


