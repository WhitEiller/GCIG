import os
import re
import json
import ast
import shutil

# 文件目录
directory = './question_outputs/'

# 获取所有txt文件
all_files = []
if os.path.exists(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            all_files.append(filepath)
else:
    print(f"目录不存在: {directory}")
    exit()

print(f"找到 {len(all_files)} 个txt文件")

# 更新问题类型列表
question_types = [
    "Single Sentence-based on sentence from Source",
    "Single Sentence-based on sentence from Source", 
    "Composite Two-sentence Integration in Source"
]

results = []


def truncate_answer_to_three_questions(answer_text: str) -> str:
    """将 **Answer** 段落在出现附加说明/后续段落前截断，只保留问题正文区域。

    截断边界：
    - 行首的 "Note:" 提示
    - 行首的 "**Source" / "**Sources" / "Source Sentences"
    - 行首的 "**Relevance**"
    取最早出现的边界位置进行截断。
    """
    boundaries = []
    for pattern in [r"(?m)^Note\s*:", r"(?m)^\*\*Source", r"(?m)^Source\s*Sentences", r"(?m)^\*\*Relevance\*\*"]:
        m = re.search(pattern, answer_text)
        if m:
            boundaries.append(m.start())
    if not boundaries:
        return answer_text
    cut = min(boundaries)
    return answer_text[:cut].rstrip()

for filepath in all_files:
    filename = os.path.basename(filepath)
    print(f"处理文件 {filename}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取sources部分
    sources_pattern = r'\*\*Sources\*\*\n(.*?)(?=\*\*Question\*\*|\*\*Answer\*\*)'
    sources_match = re.search(sources_pattern, content, re.DOTALL)
    if sources_match:
        sources_text = sources_match.group(1).strip()
        try:
            # 尝试将字符串转换为列表
            sources_list = ast.literal_eval(sources_text)
            if isinstance(sources_list, list):
                sources_text = sources_list
        except:
            # 如果转换失败，保持原始字符串
            pass
    else:
        sources_text = []
    
    # 提取**Answer**部分
    answer_pattern = r'\*\*Answer\*\*\n(.*?)(?=\Z)'
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if not answer_match:
        print(f"文件 {filename} 中没有找到**Answer**部分")
        continue
        
    answer_content = truncate_answer_to_three_questions(answer_match.group(1).strip())
    
    # 按数字序号分割问题（1. 2. 3.）
    question_pattern = r'(\d+)\.\s*\*\*(.*?)\*\*:\s*(.*?)(?=\n\d+\.\s*\*\*|\Z)'
    questions = re.findall(question_pattern, answer_content, re.DOTALL)
    # 只保留前三个问题
    questions = questions[:3]
    
    for question_number, question_type_raw, question_content in questions:
        # 清理问题类型和内容
        question_type = question_type_raw.strip()
        # 仅保留问题本身，去除多余空白
        question_clean = question_content.strip()
        
        # 为每个问题创建一个结果条目
        results.append({
            'id': f'{question_number}',  # 使用问题编号作为ID
            'type': question_type,
            'question': question_clean,
            'file_sources': sources_text,  # 文件开头的所有sources
            'filename': filename,  # 添加文件名作为参考
            'answer': ""  # 空的answer字段
        })

# 保存到json文件
output_path = 'extracted_questions_from_answer.json'
with open(output_path, 'w', encoding='utf-8') as json_file:
    json.dump(results, json_file, ensure_ascii=False, indent=2)

print(f"提取完成，共{len(results)}个问题，保存到 {output_path}")
print(f"处理了 {len(all_files)} 个文件")

# 显示一些统计信息
print("\n统计信息:")
print(f"总文件数: {len(all_files)}")
print(f"总问题数: {len(results)}")
if len(all_files) > 0:
    print(f"平均每个文件问题数: {len(results) / len(all_files):.2f}")

# # 按类型统计
# type_counts = {}
# for result in results:
#     type_name = result['type']
#     type_counts[type_name] = type_counts.get(type_name, 0) + 1

# print("\n按类型统计:")
# for type_name, count in type_counts.items():
#     print(f"{type_name}: {count}")

# # 按文件统计
# file_counts = {}
# for result in results:
#     filename = result['filename']
#     file_counts[filename] = file_counts.get(filename, 0) + 1

# print(f"\n前5个文件的问题数:")
# for i, (filename, count) in enumerate(list(file_counts.items())[:5]):
#     print(f"{filename}: {count}")

# 删除问题输出文件夹（取消注释以启用）
# try:
#     shutil.rmtree(directory)
#     print(f"已删除文件夹: {directory}")
# except Exception as e:
#     print(f"删除文件夹失败: {e}")