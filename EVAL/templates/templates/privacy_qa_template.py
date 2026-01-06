from typing import Any, Dict, List, Optional

class PrivacyQATemplate:
    """Template for Privacy QA - determining if context is relevant to question"""

    def __init__(self):
        self.name = "Privacy QA Relevance"

    def apply(self, example: Dict[str, Any]):
        """Apply template to an example"""
        question = example.get("question", "")
        text = example.get("text", "")
        answer = example.get("answer", "")

        # Create input prompt
        input_text = f"Given the context, is this related to the question?\nContext: {text}\nQuestion: {question}"

        return input_text, answer

    def get_answer_choices_list(self, example: Dict[str, Any]) -> Optional[List[str]]:
        """Return answer choices for classification task"""
        return ["Relevant", "Irrelevant"]
