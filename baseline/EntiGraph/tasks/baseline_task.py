# Rewrite from https://github.com/ZitongYang/Synthetic_Continued_Pretraining/blob/main/tasks/quality.py

import json
from hashlib import md5

from baselines.EntiGraph.tasks.task_abc import Document, Task
from baselines.EntiGraph.entigraph_utils.prompt_utils import (
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS,
                                OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS,
                                QUALITY_FEW_SHOT_COT_PROMPT, OPENAI_API_SYSTEM_QUALITY_QA_SFT)

class BaselineTask(Task):
    openai_system_generate_entities = OPENAI_API_SYSTEM_QUALITY_GENERATE_ENTITIES
    openai_system_generate_two_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_TWO_ENTITY_RELATIONS
    openai_system_generate_three_entity_relations = OPENAI_API_SYSTEM_QUALITY_GENERATE_THREE_ENTITY_RELATIONS
    openai_system_quality_qa_sft = OPENAI_API_SYSTEM_QUALITY_QA_SFT
    llama_cot_prompt = QUALITY_FEW_SHOT_COT_PROMPT

    def __init__(self, input_file: str, data_type: str):
        # Initialize for static analyzers before Task.__init__ assigns it.
        self.documents = []
        self._data = self._load_split(input_file, data_type)
        self._create_documents()
        self._dedup()

    @staticmethod
    def _load_split(input_file: str, data_type: str):
        if data_type == 'raw':
            with open(input_file, "r", encoding='utf-8') as f:
                data = []
                for line in f:
                    try:
                        data.append(json.loads(line, strict=False))
                    except Exception as e:
                        print(f"Warning: Skipping invalid JSON line: {str(e)}")
                        continue
                data = [[chunk] for chunk in data]
        elif data_type == 'chunked':
            with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Remove null bytes and other problematic characters
                content = content.replace('\x00', '')
                try:
                    data = json.loads(content, strict=False)
                except Exception as e:
                    print(f"Error loading JSON file: {str(e)}")
                    raise

        documents = []
        skipped = 0
        for doc in data:
            try:
                for chunk in doc:
                    if isinstance(chunk, dict) and 'content' in chunk:
                        # Clean the content from null bytes
                        chunk['content'] = chunk['content'].replace('\x00', '')
                        documents.append(chunk)
                    else:
                        skipped += 1
            except Exception as e:
                print(f"Warning: Skipping invalid document: {str(e)}")
                skipped += 1
                continue

        print(f"Loaded {len(documents)} documents, skipped {skipped} invalid entries")
        return documents

    def _create_documents(self):
        documents = []
        skipped = 0
        for adict in self._data:
            try:
                if 'content' in adict and adict['content']:
                    document = Document(text=adict['content'], questions=[])
                    documents.append(document)
                else:
                    skipped += 1
            except Exception as e:
                print(f"Warning: Skipping document creation: {str(e)}")
                skipped += 1
                continue

        if skipped > 0:
            print(f"Created {len(documents)} documents, skipped {skipped} invalid entries")
        super().__init__('baseline', documents)

    def _dedup(self):
        deuped_documents = {}
        for document in self.documents:
            key = compute_content_hash(document.text)
            if key not in deuped_documents:
                deuped_documents[key] = document

        self.documents = list(deuped_documents.values())


    def performance_stats(self):
        pass

    def load_attempts_json(self, file_path: str):
        pass


def compute_content_hash(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
