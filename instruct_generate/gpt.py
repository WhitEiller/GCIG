import openai
import time
import json
# API 相关配置
api_key = ""
base_url= ""
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url
)


def get_oai_completion(prompt):
    try:
        response = client.chat.completions.create(
            model="grok-4",  # 确保模型名称正确
            # model = "qwen-plus", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # 调低一点，减少随机性
            max_tokens=24000,  # 限制 token 数，避免超长输出
            top_p=0.9,  # 适当调整，增强控制
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        return response.choices[0].message.content

    except openai.OpenAIError as e:
        print(f"OpenAI API returned an error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def call_chatgpt(ins):
    success = False
    re_try_count = 45
    ans = ''

    while not success and re_try_count >= 0:
        re_try_count -= 1
        try:
            ans = get_oai_completion(ins)
            success = True
        except Exception as e:
            print(f"Error: {e}, retrying in 5 seconds...")
            time.sleep(5)

    return ans


# 测试调用
if __name__ == "__main__":
    prompt = """"
question:
answer:
"""


    response = call_chatgpt(prompt)
    print(response)
