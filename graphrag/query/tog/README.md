# ToG (Think-on-Graph) Integration

This directory contains the integrated ToG (Think-on-Graph) reasoning engine for multi-hop question answering over knowledge graphs.

## Overview

ToG provides sophisticated graph-based reasoning capabilities that have been adapted to work seamlessly with the graphrag framework.

## Usage Example

```python
from graphrag.llm import OpenAIModel
from graphrag.query.loader import load_graph
from graphrag.query.tog import ToGReasoner, ToGAdapter

# Load your knowledge graph
graph, entities, relations = load_graph(graph_dir)

# Initialize LLM
llm = OpenAIModel(
    model="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# Method 1: Using ToGReasoner directly
reasoner = ToGReasoner(
    llm=llm,
    depth=3,              # Multi-hop depth
    width=3,              # Candidate width per step
    exploration_temperature=0.4,
    reasoning_temperature=0.0
)

query = "What products are manufactured by organizations founded in California?"
reasoning_chain, answer = reasoner.reason(query, graph, entities, relations)

print(f"Answer: {answer}")
print(f"Reasoning steps: {len(reasoning_chain)}")

# Method 2: Using ToGAdapter (convenience wrapper)
reasoning_chain, answer = ToGAdapter.reason_with_graphrag(
    query=query,
    graph=graph,
    entities=entities,
    relations=relations,
    llm=llm,
    depth=3,
    width=3
)
```

## Architecture

The ToG integration consists of:

- `tog_adapter.py` - Main reasoning engine with graph exploration logic
- `../prompts/query/tog/` - Prompt templates for relation extraction, entity scoring, and answer generation
- `../../utils/tog_utils.py` - Utility functions for BM25 ranking, relation parsing, etc.

## Key Features

- **Iterative Graph Exploration**: Traverses the knowledge graph up to configurable depth
- **LLM-guided Pruning**: Uses language models to score and filter relevant relations/entities
- **Multi-hop Reasoning**: Builds reasoning chains across multiple hops in the graph
- **Sufficiency Evaluation**: Determines when enough information has been gathered
- **Transparent Reasoning**: Returns full reasoning chain for interpretability

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `depth` | 3 | Maximum depth for graph exploration |
| `width` | 3 | Number of top candidates to keep at each step |
| `exploration_temperature` | 0.4 | Temperature for exploration phase (higher = more creative) |
| `reasoning_temperature` | 0.0 | Temperature for reasoning phase (0 = deterministic) |
| `max_tokens` | 256 | Maximum tokens for LLM generation |

## Optional Dependencies

For full functionality, install:

```bash
pip install rank-bm25 sentence-transformers
```

- `rank-bm25`: For BM25-based relation ranking
- `sentence-transformers`: For semantic similarity-based pruning

## Reference

If you use ToG in your research, please cite:

```bibtex
@inproceedings{sun2024thinkongraph,
  title={Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph},
  author={Sun, Jiashuo and Xu, Chengjin and Tang, Lumingyuan and Wang, Saizhuo and Lin, Chen and Gong, Yeyun and Shum, Heung-Yeung and Guo, Jian},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

**Original Repository:** https://github.com/GasolSun36/ToG