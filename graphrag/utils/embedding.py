import os
import requests
import json

from openai import OpenAI


def get_embedding(text: str | list[str]):
    client = OpenAI(
        api_key="123",
        base_url = "http://10.4.177.241:9997/v1"
    )
    if isinstance(text, str):
        text = [text]
    MAX_BATCH = 10
    results = []
    for i in range(0, len(text), MAX_BATCH):
        completion = client.embeddings.create(
            # model="text-embedding-v3",
            model = "bge-m3",
            input=text[i: i + MAX_BATCH],
            # dimensions=1024,  # 模型不支持指定维度，注释掉
            encoding_format="float"
        )
        results.extend([data.embedding for data in completion.data])
    return results

