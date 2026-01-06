ENTITY_ALIGNMENT_PROMPT = """You are given a list of entities, each with a name and description. Your task is to group same entities that refer to the same real-world entity.
**Note**: Return the result in the following JSON format:
[[<entity_name>, <entity_name>, ...], ...]

##### Example #####
Entity List:
[
    {{
        "name": "China",
        "description": "China, the world's most populous country in East Asia, boasts a millennia-old civilization."
    }},
    {{
        "name": "People's Republic of China",
        "description": "The People's Republic of China is a socialist state in East Asia.",
    }},
    {{
        "name": "US",
        "description": "The United States (US), a federal republic in North America, comprises 50 states and a federal district."
    }}
]

Output:
[["Chain", "People's Republic of China"], ["US"]]

##### Real Data #####
Entity List:
{entities}

Output:
"""