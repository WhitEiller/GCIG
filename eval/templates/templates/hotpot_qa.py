from typing import Any, Dict, List, Optional

class HotpotQATemplate:
    """Template for HotpotQA question answering without context"""
    
    def __init__(self):
        self.name = "HotpotQA Direct QA"
    
    def apply(self, example: Dict[str, Any]):
        """Apply template to an example"""
        question = example.get("question", "")
        answer = example.get("answer", "")
        
        # Create input prompt - direct question answering
        input_text = question
        
        return input_text, answer
    
    def get_answer_choices_list(self, example: Dict[str, Any]) -> Optional[List[str]]:
        """For generation tasks, return None to indicate it's not multiple choice"""
        return None  # This is a generation task, not multiple choice