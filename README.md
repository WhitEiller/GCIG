# GCIG: GraphRAG-based Cross-document Instruction Generation


Official implementation of the paper: **"GCIG: GraphRAG-based Cross-document Instruction Generation for Boosting LLM Reasoning"**

<p align="center">
  <img src="resources/1.png" width="90%" alt="GCIG Framework Overview"/>
</p>

## Overview

GCIG is a framework for automatically generating high-quality cross-document instruction data to enhance LLMs' complex reasoning capabilities. Unlike existing methods that rely on isolated documents, GCIG leverages GraphRAG to integrate knowledge across documents and generate instruction data with better logical coherence and broader knowledge coverage.

### Key Features

- **Enhanced Knowledge Graph Construction**: Builds entity-centric graphs linking document evidence with entities
- **LLM-driven Chunk Selection (LCS)**: Forms multi-hop evidence chains with factual consistency
- **Adaptive Prompt-based Question Generation (APQG)**: Produces diverse multi-hop questions
- **Chain-of-Thought-guided Answer Generation (CoTAG)**: Generates interpretable answers with logical depth

<p align="center">
  <img src="resources/2.png" width="100%" alt="GCIG Architecture"/>
</p>

## Project Structure

```
GCIG/
├── EKG_construct/              # Enhanced Knowledge Graph construction module
│   ├── graph_mult_construct.py # Multi-threaded entity extraction
│   ├── graph_summary.py        # LLM-based summarization & embedding
│   ├── graph_align.py          # Entity alignment & disambiguation
│   ├── graph_connect.py        # Text unit connection
│   ├── config.py               # Configuration management
│   └── construct.sh            # Pipeline execution script
│
├── graphrag/                   # Core GraphRAG framework with ToG integration
│   ├── index/                  # Graph construction pipeline
│   │   ├── components/         # GraphExtractor, Aligner, Connector
│   │   └── text_split.py       # Token-based text chunking
│   ├── llm/                    # Language model interfaces
│   │   ├── base_model.py       # Abstract LLM interface
│   │   └── openai_model.py     # OpenAI API implementation
│   ├── model/                  # Data models (Entity, Relation, TextUnit)
│   ├── prompts/                # LLM prompt templates
│   │   ├── index/              # Extraction, alignment, summarization
│   │   └── query/              # Entity extraction, answer generation
│   ├── query/                  # Query processing & reasoning
│   │   ├── retrieval.py        # Seed entity & subgraph retrieval
│   │   ├── query.py            # Query execution
│   │   └── loader.py           # Data loading utilities
│   └── utils/                  # Embedding, retrieval, transformation
│
├── ToG/                        # Think-on-Graph: Deep reasoning on KG (ICLR 2024)
│   ├── ToG/                    # Core ToG algorithm implementation
│   │   ├── main_freebase.py    # ToG with Freebase backend
│   │   ├── main_wiki.py        # ToG with Wikidata backend
│   │   ├── freebase_func.py    # Relation search, entity pruning, reasoning
│   │   ├── wiki_func.py        # Wikidata-specific operations
│   │   ├── utils.py            # LLM interface, BM25/SentenceBERT ranking
│   │   └── prompt_list.py      # Pruning, scoring, reasoning prompts
│   ├── CoT/                    # Chain-of-Thought baseline methods
│   ├── data/                   # 9 KBQA benchmark datasets
│   ├── eval/                   # Exact Match evaluation
│   ├── Freebase/               # Freebase KG configuration
│   └── Wikidata/               # Wikidata KG database & deployment
│
├── GraphGen/                   # Synthetic data generation framework
│   └── ...                     # (see GraphGen/README.md)
│
└── baselines/                  # Baseline comparison methods
    ├── self-instruct/
    ├── bonito/
    ├── rag-instruct/
    └── entigraph/
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/WhitEiller/GCIG.git
cd GCIG

# Create virtual environment
conda create -n gcig python=3.10
conda activate gcig

# Install dependencies
pip install -r requirements.txt

# Install GCIG
cd GCIG
pip install -e .
cd ..
```

## Quick Start

### 1. Build Enhanced Knowledge Graph

```python
from EKG_Instruct.GC.graph_mutl_construct import build_enhanced_kg

# Build knowledge graph from corpus
kg = build_enhanced_kg(
    corpus_path="path/to/your/corpus",
    output_path="path/to/output/kg"
)
```

### 2. Generate Instructions using GraphGen

```bash
cd GraphGen

# Configure your API key
cp .env.example .env
# Edit .env to add your API key

# Run instruction generation
python -m graphgen.generate \
    --input_path path/to/corpus \
    --output_path path/to/output \
    --model gpt-4
```

### 3. Evaluate Generated Data

```bash
cd nayak-aclfindings24-code

# Run evaluation
bash run_eval_all_models.sh
```

## Experimental Results

<p align="center">
  <img src="resources/4.png" width="90%" alt="GCIG Framework Overview"/>
</p>

## Components

### GraphGen

A framework for synthetic data generation guided by knowledge graphs. Supports:
- Multiple LLM inference backends (OpenAI, Ollama, vLLM, SGLang)
- Various data sources (files, databases, search engines)
- Multiple output formats (Atomic, Aggregated, CoT, Multi-hop, VQA)

See [GraphGen README](GraphGen/README.md) for detailed documentation.

### EKG-Instruct

Enhanced Knowledge Graph based instruction generation:
- `GC/`: Graph construction with entity extraction and alignment
- `generate/`: Question and answer generation with adaptive prompts

### Baselines

We compare against several baseline methods:
- Self-Instruct
- Bonito
- RAG-Instruct
- EntiGraph

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# API Configuration
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Model Configuration
DEFAULT_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{gcig2025,
  title={GCIG: GraphRAG-based Cross-document Instruction Generation for Boosting LLM Reasoning},
  author={},
  booktitle={},
  year={2025}
}
```

## ToG Integration

### Think-on-Graph: Deep Reasoning on Knowledge Graphs

We have integrated **ToG (Think-on-Graph)** [[ICLR 2024](https://arxiv.org/abs/2307.07697)] into our framework to enhance multi-hop reasoning capabilities. ToG provides sophisticated graph-based reasoning that complements GCIG's cross-document instruction generation.

#### ToG Core Capabilities

**Algorithm Pipeline:**
1. **Entity Linking** - Identifies topic entities from natural language questions
2. **Iterative Relation Exploration** - Discovers relevant relations at configurable depth levels (default: 3)
3. **Entity Search & Pruning** - Finds and filters candidate entities using:
   - LLM-based semantic pruning
   - BM25 lexical matching
   - SentenceBERT embedding-based ranking
4. **Reasoning & Answer Generation** - Determines information sufficiency and generates answers with reasoning chains

**Key Features:**
- Multi-hop reasoning over knowledge graphs (Freebase/Wikidata backends)
- Temperature-controlled exploration (0.4) vs. reasoning (0.0)
- Fallback to Chain-of-Thought for complex queries
- Exact Match evaluation on 9 KBQA benchmarks (CWQ, WebQSP, GrailQA, QALD-10, etc.)

#### Integration Architecture

ToG's reasoning capabilities are integrated with the `graphrag/` module:

```python
# Example: Using ToG-enhanced reasoning in graphrag
from graphrag.query import generate
from graphrag.query.tog_engine import ToGReasoner

# Standard GraphRAG query
context, answer = generate(
    query="What products are manufactured by organizations founded in California?",
    graph=kg,
    entities=entities,
    relations=relations
)

# ToG-enhanced multi-hop reasoning
tog_reasoner = ToGReasoner(
    kg_backend="custom",  # Uses graphrag's internal KG
    depth=3,              # Multi-hop depth
    width=3,              # Candidate width per step
    pruning_method="llm"  # LLM-based relevance scoring
)
reasoning_chain, answer = tog_reasoner.reason(query, kg)
```

**Integration Points:**
- **LLM Interface:** ToG's `run_llm()` reuses `graphrag.llm.OpenAIModel`
- **Data Models:** Entity/Relation structures align with `graphrag.model.*`
- **Prompts:** ToG prompt templates integrated into `graphrag.prompts.query.tog/`
- **Reasoning Engine:** `graphrag.query.tog_engine` provides iterative graph exploration
- **Evaluation:** ToG's Exact Match metrics available in `ToG/eval/`

#### Citation

If you use ToG components in your research:

```bibtex
@inproceedings{sun2024thinkongraph,
  title={Think-on-Graph: Deep and Responsible Reasoning of Large Language Model with Knowledge Graph},
  author={Sun, Jiashuo and Xu, Chengjin and Tang, Lumingyuan and Wang, Saizhuo and Lin, Chen and Gong, Yeyun and Shum, Heung-Yeung and Guo, Jian},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

**Original Repository:** [ToG on GitHub](https://github.com/GasolSun36/ToG)

---

## Acknowledgements

This project builds upon several excellent open-source projects:
- [**ToG**](https://github.com/GasolSun36/ToG) - Think-on-Graph for deep reasoning on knowledge graphs (ICLR 2024)
- [**GraphGen**](https://github.com/open-sciencelab/GraphGen) - Synthetic data generation framework
- [**GraphRAG**](https://github.com/microsoft/graphrag) - Microsoft's graph-based retrieval-augmented generation
- [**Bonito**](https://github.com/BatsResearch/bonito) - Conditional task generation framework
- [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory) - Efficient LLM fine-tuning toolkit

We are particularly grateful to the ToG team for their pioneering work on LLM reasoning over knowledge graphs, which has been instrumental in enhancing GCIG's multi-hop reasoning capabilities.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and feedback, please open an issue on GitHub.
