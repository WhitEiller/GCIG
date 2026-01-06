#!/usr/bin/env python3
import json

def merge_datasets():
    # 读取第一个数据集 (cleaned_answer_9138.json)
    print("Loading first dataset...")
    with open('data/cleaned_answer_9138.json', 'r') as f:
        dataset1 = json.load(f)
    print(f"First dataset loaded: {len(dataset1)} items")
    
    # 读取第二个数据集 (cleaned_data.json)  
    print("Loading second dataset...")
    with open('data/cleaned_data.json', 'r') as f:
        dataset2 = json.load(f)
    print(f"Second dataset loaded: {len(dataset2)} items")
    
    # 合并数据集
    merged_data = []
    
    # 从第一个数据集提取 question 和 answer
    print("Processing first dataset...")
    for item in dataset1:
        if 'question' in item and 'answer' in item:
            merged_data.append({
                'question': item['question'],
                'answer': item['answer']
            })
    print(f"Added {len([item for item in dataset1 if 'question' in item and 'answer' in item])} items from first dataset")
    
    # 从第二个数据集提取 question 和 output (作为 answer)
    print("Processing second dataset...")
    count_second = 0
    for item in dataset2:
        if 'question' in item and 'output' in item:
            merged_data.append({
                'question': item['question'],
                'answer': item['output']
            })
            count_second += 1
    print(f"Added {count_second} items from second dataset")
    
    print(f"Total merged items: {len(merged_data)}")
    
    # 保存合并后的数据集
    print("Saving merged dataset...")
    with open('data/merged_qa_dataset.json', 'w') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged dataset saved to data/merged_qa_dataset.json")
    
    # 显示一些统计信息
    print("\nSample from merged dataset:")
    for i in range(min(3, len(merged_data))):
        print(f"Item {i+1}:")
        print(f"  Question: {merged_data[i]['question'][:100]}...")
        print(f"  Answer: {merged_data[i]['answer']}")
        print()

if __name__ == "__main__":
    merge_datasets()
