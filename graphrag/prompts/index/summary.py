ENTITY_DESC_SUMMARAY_PROMPT = """Based on the provided text, generate a summary of {entity}.
**NOTE**:
1. The summary must not exceed 70 words.
2. Only output the generated summary content without any additional explanations.

Text:
{text}

Output:
"""

RELATION_DESC_SUMMARY_PROMPT = """Based on the provided text, generate a summary of the relationship between {source} and {target}.
**NOTE**:
1. The summary must not exceed 70 words.
2. Only output the generated summary content without any additional explanations.

Text:
{text}

Output:
"""