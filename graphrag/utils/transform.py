import json
import re


def str2json(text: str):
    # 处理空字符串或None
    if not text or not text.strip():
        return {"entities": [], "relations": []}
    
    # 删除markdown代码块标记
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    # 寻找JSON开始和结束位置
    start = 0
    end = len(text) - 1
    
    # 安全查找开始位置
    while start < len(text) and text[start] not in ("{", "["):
        start += 1
    
    # 安全查找结束位置  
    while end >= 0 and text[end] not in ("}", "]"):
        end -= 1
    
    # 如果找不到有效的JSON结构，返回空结果
    if start >= len(text) or end < 0 or start > end:
        return {"entities": [], "relations": []}
    
    # 提取JSON部分
    json_text = text[start:end+1]
    
    # 删除注释
    json_text = re.sub(r"//.*?\n", "\n", json_text)
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # 如果JSON解析失败，尝试修复常见问题
        try:
            # 移除可能的尾随逗号
            fixed_text = re.sub(r',\s*([}\]])', r'\1', json_text)
            return json.loads(fixed_text)
        except:
            # 最后的fallback：返回空结果（静默处理，不显示详细信息）
            return {"entities": [], "relations": []}