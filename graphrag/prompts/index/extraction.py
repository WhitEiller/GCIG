GRAPH_EXTRACTION_PROMPT="""Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in text
- entity_type: One of the following types: {entity_types}
- entity_description: Comprehensive description of the entity
Format each entity as {{"name", <entity_name>, "type": <entity_type>, "description": <entity_description>}}

2. From the entities identified in **Step 1**, identify all pairs of (source_entity, target_entity) that are **related** to each other.
For each pair of related entities, extract the following information:
- source: name of source entity, as identified in step 1
- target: name of target entity, as identified in step 1
- relation_description: description of the relation between source entity and target entity
Format each relations as{{"source": <source_entity (from given entity list)>, "target": <target_entity (from given entity list)>, "description": <relation_description>}}

3. Return ONLY the JSON result without any explanation or additional text:
{{
    "entities": [
        {{
            "name": <entity_name>,
            "type": <entity_type>,
            "description": <entity_description>
        }},
        ...
    ],
    "relations": [
        {{
            "source": <source_entity (from given entity list)>,
            "target": <target_entity (from given entity list)>,
            "description": <relation_description>
        }},
        ...
    ]
}}

-Real Data-
entity_types:
{entity_types}

text:
{input_text}

IMPORTANT: Respond with ONLY the JSON format above. Do not include any explanations, comments, or additional text.

output:
"""

LOOP_PROMPT = "Answer YES | NO if there are still entity or relations that need to be added."

CONTINUE_PROMPT = "Continue extracting entities and relations. Respond with ONLY the JSON format."