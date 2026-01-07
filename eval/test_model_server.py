#!/usr/bin/env python3
"""
Test script for the model server
"""
import requests
import json

def test_model_server(base_url="http://localhost:9989"):
    """Test the model server endpoints"""

    print("Testing model server...")

    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
        else:
            print("✗ Health check failed")
            return
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return

    # Test model info
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            print("✓ Model info endpoint working")
            info = response.json()
            print(f"  Model name: {info['model_name']}")
            print(f"  Base model: {info['base_model']}")
            print(f"  Checkpoint: {info['checkpoint_path']}")
        else:
            print("✗ Model info failed")
    except Exception as e:
        print(f"✗ Model info failed: {e}")

    # Test text generation
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature."
    ]

    for prompt in test_prompts:
        try:
            data = {
                "prompt": prompt,
                "max_new_tokens": 128,
                "temperature": 0.1,
                "do_sample": False
            }

            response = requests.post(f"{base_url}/generate", json=data)

            if response.status_code == 200:
                result = response.json()
                print(f"✓ Generation successful for: {prompt[:50]}...")
                print(f"  Response: {result['response'][:100]}...")
            else:
                print(f"✗ Generation failed for: {prompt[:50]}...")
                print(f"  Error: {response.text}")

        except Exception as e:
            print(f"✗ Generation failed for: {prompt[:50]}... Error: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:9989"

    test_model_server(base_url)