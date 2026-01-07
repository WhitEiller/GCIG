#!/usr/bin/env python3
"""
Test OpenAI API compatibility
"""
import requests
import json

def test_openai_api(base_url="http://localhost:9989"):
    """Test OpenAI API compatible endpoints"""

    print("Testing OpenAI API compatibility...")

    # Test /v1/models endpoint
    try:
        response = requests.get(f"{base_url}/v1/models")
        if response.status_code == 200:
            print("✓ /v1/models endpoint working")
            models = response.json()
            print(f"  Available models: {[model['id'] for model in models['data']]}")
        else:
            print("✗ /v1/models endpoint failed")
            return
    except Exception as e:
        print(f"✗ /v1/models endpoint failed: {e}")
        return

    # Test /v1/chat/completions endpoint
    try:
        data = {
            "model": "bonito",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }

        response = requests.post(f"{base_url}/v1/chat/completions", json=data)

        if response.status_code == 200:
            result = response.json()
            print("✓ /v1/chat/completions endpoint working")
            print(f"  Response: {result['choices'][0]['message']['content'][:100]}...")
            print(f"  Usage: {result['usage']}")
        else:
            print("✗ /v1/chat/completions endpoint failed")
            print(f"  Error: {response.text}")

    except Exception as e:
        print(f"✗ /v1/chat/completions endpoint failed: {e}")

    # Test /v1/completions endpoint
    try:
        data = {
            "model": "bonito",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "temperature": 0.1
        }

        response = requests.post(f"{base_url}/v1/completions", json=data)

        if response.status_code == 200:
            result = response.json()
            print("✓ /v1/completions endpoint working")
            print(f"  Response: {result['choices'][0]['text']}")
            print(f"  Usage: {result['usage']}")
        else:
            print("✗ /v1/completions endpoint failed")
            print(f"  Error: {response.text}")

    except Exception as e:
        print(f"✗ /v1/completions endpoint failed: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:9989"

    test_openai_api(base_url)