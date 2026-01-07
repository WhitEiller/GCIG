import re
import json
import os

def extract_json_from_log(log_file_path, output_prefix="answer"):
    """
    从日志文件中提取问题和答案数据，生成JSON格式
    每10000条数据保存一个文件
    """
    results = []
    batch_size = 10000
    file_counter = 0
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式分割每个条目
        entries = re.split(r'={80}', content)
        
        for entry in entries:
            if not entry.strip():
                continue
                
            # 提取问题ID
            id_match = re.search(r'问题ID:\s*(.+?)(?=\n)', entry)
            if not id_match:
                continue
            question_id = id_match.group(1).strip()
            
            # 提取问题类型
            type_match = re.search(r'Question Type:\s*(.+?)(?=\n)', entry)
            if not type_match:
                continue
            question_type = type_match.group(1).strip()
            
            # 提取问题内容
            question_match = re.search(r'Question:\s*(.+?)(?=\n\n|Please provide)', entry, re.DOTALL)
            if not question_match:
                continue
            question = question_match.group(1).strip()
            
            # 提取Context内容（file_sources）
            context_match = re.search(r'Context:\n(.*?)(?=\n\nQuestion Type:)', entry, re.DOTALL)
            if context_match:
                context_content = context_match.group(1).strip()
                # 将Context内容按行分割成列表
                file_sources = [line.strip() for line in context_content.split('\n') if line.strip()]
            else:
                file_sources = []
            
            # 提取GPT回答
            answer_match = re.search(r'GPT回答:\n(.*?)(?=\n\n处理耗时|\n\n={80}|\Z)', entry, re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                answer = ""
            
            # 过滤掉无效的回答
            if answer == "无回答或回答为空" or answer.startswith("处理出错:"):
                answer = ""
            
            # 创建数据条目
            data_entry = {
                'id': question_id,
                'type': question_type,
                'question': question,
                'file_sources': file_sources,
                'answer': answer
            }
            
            results.append(data_entry)
            
            # 每10000条数据保存一次
            if len(results) >= batch_size:
                file_counter += batch_size
                output_filename = f"{output_prefix}_{file_counter}.json"
                
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"已保存 {len(results)} 条数据到 {output_filename}")
                results = []  # 清空结果列表
        
        # 保存剩余的数据
        if results:
            file_counter += len(results)
            output_filename = f"{output_prefix}_{file_counter}.json"
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"已保存最后 {len(results)} 条数据到 {output_filename}")
            
        print(f"总共提取 {file_counter} 条数据")
        return file_counter
        
    except Exception as e:
        print(f"提取过程中出错: {str(e)}")
        return 0

def validate_extracted_data(data):
    """
    验证提取的数据质量
    """
    print(f"\n=== 数据质量报告 ===")
    print(f"总条目数: {len(data)}")
    
    if not data:
        return
    
    # 统计各字段的完整性
    complete_ids = sum(1 for item in data if item.get('id'))
    complete_types = sum(1 for item in data if item.get('type'))
    complete_questions = sum(1 for item in data if item.get('question'))
    complete_sources = sum(1 for item in data if item.get('file_sources'))
    complete_answers = sum(1 for item in data if item.get('answer'))
    
    print(f"完整ID数: {complete_ids}")
    print(f"完整类型数: {complete_types}")
    print(f"完整问题数: {complete_questions}")
    print(f"完整源文件数: {complete_sources}")
    print(f"完整答案数: {complete_answers}")
    
    # 统计问题类型分布
    type_counts = {}
    for item in data:
        q_type = item.get('type', 'Unknown')
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    
    print(f"\n问题类型分布:")
    for q_type, count in sorted(type_counts.items()):
        print(f"  {q_type}: {count}")
    
    # 显示一些示例数据
    print(f"\n=== 示例数据 ===")
    for i, item in enumerate(data[:2]):  # 显示前2个条目
        print(f"\n条目 {i+1}:")
        print(f"  ID: {item.get('id', 'N/A')}")
        print(f"  Type: {item.get('type', 'N/A')}")
        print(f"  Question: {item.get('question', 'N/A')[:100]}...")
        print(f"  Sources count: {len(item.get('file_sources', []))}")
        print(f"  Answer: {item.get('answer', 'N/A')[:100]}...")

def load_and_validate_all_files(output_prefix="answer"):
    """
    加载所有生成的JSON文件并进行验证
    """
    import glob
    
    # 查找所有生成的文件
    pattern = f"{output_prefix}_*.json"
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not files:
        print("没有找到生成的JSON文件")
        return []
    
    all_data = []
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"加载了 {len(data)} 条数据从 {file_path}")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
    
    print(f"\n总共加载了 {len(all_data)} 条数据")
    return all_data

if __name__ == "__main__":
    # 配置文件路径
    log_file_path = './gpt_interaction_log.txt'
    output_prefix = 'answer'
    
    # 检查日志文件是否存在
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件 {log_file_path} 不存在")
        exit(1)
    
    print(f"开始从日志文件提取数据...")
    print(f"输入文件: {log_file_path}")
    print(f"输出文件格式: {output_prefix}_10000.json, {output_prefix}_20000.json, ...")
    
    # 提取数据
    total_extracted = extract_json_from_log(log_file_path, output_prefix)
    
    if total_extracted > 0:
        print(f"\n提取完成！总共提取了 {total_extracted} 条数据")
        
        # 加载所有文件并验证数据质量
        print("\n开始验证数据质量...")
        all_data = load_and_validate_all_files(output_prefix)
        
        if all_data:
            validate_extracted_data(all_data)
    else:
        print("没有提取到任何数据")
    
    print(f"\n处理完成！") 