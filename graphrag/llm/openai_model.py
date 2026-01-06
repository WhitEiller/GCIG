import logging
import time

from openai import OpenAI

from .base_model import LLM


class OpenAIModel(LLM):
    def __init__(self,
                 api_key: str,
                 model: str,
                 base_url: str = None,
                 stream: bool = False):
        super().__init__()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.stream = stream
        self._sys_prompt = {"role": "system", "content": "You are a helpful assistant that responds only with valid JSON format. Never include explanations or additional text outside the JSON."}
        self.reset()
    
    def _generate(self, input: str, messages: list[dict[str, str]] = None) -> str:
        messages.append({"role": "user", "content": input})
        
        # 添加额外的JSON格式提醒
        messages.append({"role": "assistant", "content": "I will respond with only valid JSON format."})
        messages.append({"role": "user", "content": "Please provide your response:"})
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,  # 设置为0以获得更一致的输出
                    max_tokens=2048,  # 限制输出长度
                    top_p=1.0,       # 确定性输出
                    # response_format={"type": "json_object"},  # 如果模型支持，强制JSON格式
                    stream=self.stream
                )
                
                if self.stream:
                    result = ""
                    for chunk in response:
                        if chunk.choices:
                            result += chunk.choices[0].delta.content
                else:
                    result = response.choices[0].message.content
                
                # 基本验证：检查响应是否包含基本的JSON结构
                if result and ('{' in result or '[' in result):
                    return result
                else:
                    raise ValueError(f"LLM返回无效格式: {result[:100]}")
                
            except Exception as e:
                retry_count += 1
                print(f"LLM调用失败 (尝试 {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    # 返回一个空的JSON结构作为fallback
                    return '{"entities": [], "relations": []}'
                
                time.sleep(2 ** retry_count)  # 指数退避
        
        return '{"entities": [], "relations": []}'