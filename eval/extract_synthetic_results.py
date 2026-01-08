#!/usr/bin/env python3
import json
import re
from collections import Counter

def clean_json_string(text):
    # Remove or escape problematic control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text

def load_and_clean_dataset(file_path):
    """Load JSONL format data (one JSON object per line)"""
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # Clean the line
            line = clean_json_string(line)
            
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON decode error on line {line_num}: {e}")
                continue
    
    return data

def extract_context_from_input(input_text):
    """Extract the passage/context from the input text, removing the first word"""
    # Find "Based on that information" or "Based on the previous passage" to split context from question
    patterns = [
        r"^(.*?)\s+Based on (?:that information|the previous passage)",
        r"^(.*?)\s+is it true that"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
        if match:
            context = match.group(1).strip()
            # Remove the first word (usually the person's name or entity name)
            words = context.split()
            if len(words) > 1:
                context = ' '.join(words[1:])
            return context
    
    # Fallback: try to find the context before question patterns
    lines = input_text.split('\n')
    context_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith('based on'):
            context_lines.append(line)
        else:
            break
    
    context = ' '.join(context_lines).strip()
    # Remove the first word
    words = context.split()
    if len(words) > 1:
        context = ' '.join(words[1:])
    
    return context

def extract_question_from_input(input_text):
    """Extract the question from the input text"""
    # Look for "Based on that information" pattern
    pattern = r"(Based on (?:that information|the previous passage),.*?)$"
    match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Look for other question patterns
    patterns = [
        r"(is it true that.*?\?)",
        r"(.*\? Yes, no, or maybe\?)$",
        r"(.*true, false, or inconclusive\?)$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return ""

def convert_synthetic_results_to_qa_format(data):
    """Convert synthetic_results.json format to bio_qa format"""
    converted_data = []
    
    for item in data:
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        # Extract context and question from input
        context = extract_context_from_input(input_text)
        question = extract_question_from_input(input_text)
        
        # Use the output as the answer
        answer = output
        
        converted_item = {
            "question": question,
            "answer": answer,
            "context": context
        }
        
        converted_data.append(converted_item)
    
    return converted_data

if __name__ == "__main__":
    # Process the synthetic_results.json file
    source_file = "//test1/bonito2/synthetic_results.json"
    output_file = "data/synthetic_results_clean.json"
    
    print(f"Loading data from {source_file}...")
    data = load_and_clean_dataset(source_file)
    
    if data:
        print(f"Loaded {len(data)} items")
        
        # Convert format
        print("Converting to QA format...")
        converted_data = convert_synthetic_results_to_qa_format(data)
        
        # Extract unique questions for analysis
        questions = set()
        contexts = set()
        for item in converted_data:
            questions.add(item['question'].strip())
            contexts.add(item['context'][:100] + "..." if len(item['context']) > 100 else item['context'])
        
        print(f"Found {len(questions)} unique questions")
        print(f"Found {len(contexts)} unique contexts")
        
        # Show first 3 converted items with full format
        print("\n" + "="*80)
        print("FIRST 3 EXTRACTED ITEMS:")
        print("="*80)
        for i, item in enumerate(converted_data[:3], 1):
            print(f"\n--- ITEM {i} ---")
            print(f"Context: {item['context']}")
            print(f"Question: {item['question']}")
            print(f"Answer: {item['answer']}")
            print("-" * 50)
        
        # Show sample questions
        print("\nSample unique questions:")
        for question in sorted(list(questions)[:5]):
            print(f"  - {question}")
        
        # Show sample contexts
        print("\nSample unique contexts:")
        for context in sorted(list(contexts)[:3]):
            print(f"  - {context}")
        
        # Save cleaned data
        with open(output_file, "w") as f:
            json.dump(converted_data, f, indent=2)
        print(f"\nSaved {len(converted_data)} converted items to {output_file}")
    else:
        print("Failed to load or clean the data")