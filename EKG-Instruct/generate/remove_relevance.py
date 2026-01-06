import os
import re

def remove_relevance_section(content):
    """Remove the Relevance section and everything after it from the content."""
    # Find the position of "**Relevance**" (case insensitive)
    pattern = r'\*\*Relevance\*\*.*'
    # Remove everything from "**Relevance**" to the end of the file
    cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_content.rstrip()

def process_files(directory):
    """Process all .txt files in the directory to remove Relevance sections."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return
    
    processed_count = 0
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            
            try:
                # Read the file
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove relevance section
                cleaned_content = remove_relevance_section(content)
                
                # Write back only if content changed
                if cleaned_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    print(f"Processed: {filename}")
                    processed_count += 1
                else:
                    print(f"No relevance section found in: {filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    print(f"\nTotal files processed: {processed_count}")

if __name__ == "__main__":
    # Process files in the question_outputs directory
    directory = "question_outputs"
    process_files(directory)