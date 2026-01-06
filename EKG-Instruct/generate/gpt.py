import openai
import time
import json
# API 相关配置
api_key = "sk-QknoYx5sDbPI0FofwPrguh8pbgkA4tj2TDT2Zmtmn7xD4poy"
base_url="http://10.4.177.69:9997/v1"
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
['Recipient will, and will cause its Representatives to  (i) hold the Confidential Information in strict confidence, in a manner consistent with the protections it employs to protect its own confidential information of a similar nature, and in any event no less than a reasonable standard of care and in strict accordance with the provisions of this Agreement; and  (ii) use the Confidential Information of the Disclosing Party for no purpose other than the Permitted Use.', '3. The Recipient undertakes to keep the Confidential Information secure and not to disclose it to any third party [except to its employees [and professional advisers] who need to know the same for the Purpose, who know they owe a duty of confidence to the Discloser and who are bound by obligations equivalent to those in clause 2 above and this clause 3.', '3. The Recipient undertakes to keep the Confidential Information secure and not to disclose it to any third party [except to its employees [and professional advisers] who need to know the same for the Purpose, who know they owe a duty of confidence to the Discloser and who are bound by obligations equivalent to those in clause 2 above and this clause 3.']
['Recipient: The entity receiving confidential information, potentially bound by confidentiality agreements.', 'Disclosing Party: The Disclosing Party shares confidential information with the Recipient, seeks legal protection for it, and provides Confidential Information.', 'Discloser: Discloser is the entity that provides, owns, and designates Confidential Information to the Recipient, often under an agreement, with rights to request its return or destruction.', 'Agreement: A legal contract between the Company and FNHA outlining terms for handling Confidential Information.', 'employees: Employees from various parties, EFCA, Provider, and Recipient who may access or receive confidential information as needed for their work, bound by confidentiality obligations.', 'Confidential Information: Confidential Information includes proprietary and sensitive data disclosed under agreements, such as research, product plans, customer lists, and technical details, which must be kept secure and not disclosed to unauthorized parties.', 'third party: A third party is an external entity not authorized to receive confidential information from EFCA or the Provider without consent, not bound by confidentiality agreements, and may independently provide information to the Receiving Party.', 'Purpose: The reason or objective for which the confidential information is shared.', 'CONFIDENTIAL INFORMATION: Proprietary specifications, designs, plans, drawings, and business/technical information protected under the agreement with restrictions on use and disclosure.', 'Representatives: Representatives include affiliates, officers, directors, employees, consultants, advisers, and entities or individuals who may access confidential information for business evaluations, agreements, or from PHCS.', 'Permitted Use: Confidential information may be used for research, product development, and evaluating or pursuing potential business relationships with the Disclosing Party or its affiliates.', 'regulation: Regulations may mandate or require the disclosure of confidential information by recipients.', 'professional advisers: Professional advisers, including experts and consultants, employed by or working with the Recipient, who need access to Confidential Information for specific purposes and are bound by confidentiality obligations.', 'Party: A party in the agreement, responsible for notice of termination, maintaining obligations, handling Proprietary Information, claiming interests, and representing entities or organizations, bound to confidentiality.']
Based on the above knowledge, determine the following questions. And make a brief and clear response.
Which combination of statements best encapsulates the recipient’s obligations regarding the use, reproduction, and disclosure of confidential information?
answer:
"""


    response = call_chatgpt(prompt)
    print(response)
