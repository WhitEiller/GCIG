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
│   ├── graph_mult_construct.py # Stage 1: Multi-threaded entity extraction
│   ├── graph_summary.py        # Stage 2: LLM-based summarization & embedding
│   ├── graph_align.py          # Stage 3: Entity alignment & disambiguation
│   ├── graph_connect.py        # Stage 4: Text unit connection
│   ├── config.py               # Configuration management
│   ├── construct.sh            # Pipeline execution script
│   └── README.MD               # Detailed module documentation
│
├── instruct_generate/          # Cross-document instruction generation module
│   ├── extract_entity.py       # Stage 1: LCS - Chunk selection & multi-hop retrieval
│   ├── question.py             # Stage 2: APQG - Question generation (6 types)
│   ├── relevance.py            # Stage 3: Cross-document relevance assessment
│   ├── filtered.py             # Stage 4: Quality filtering & ranking
│   ├── answer.py               # Stage 5: CoTAG - GraphRAG answer generation
│   └── README.md               # Detailed module documentation
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

## Dependencies

### Core Dependencies

```
# Graph Processing
networkx>=2.6           # Graph data structures and algorithms
pyarrow>=8.0            # Parquet file I/O for efficient storage
pandas>=1.3             # DataFrame operations

# Language Models
openai>=1.0             # OpenAI API client for LLM interactions
graphrag                # Core GraphRAG framework

# Embeddings & Retrieval
sentence-transformers   # Semantic similarity and embeddings
rank_bm25>=0.2         # BM25 lexical retrieval

# Deep Learning (Optional, for GPU acceleration)
torch>=2.0              # PyTorch for embedding models
transformers>=4.30      # Hugging Face transformers

# Utilities
tqdm>=4.62             # Progress visualization
python-dotenv          # Environment variable management
```

For complete dependency lists:
- EKG Construction: See `EKG_construct/requirements.txt`
- Instruction Generation: See `instruct_generate/requirements.txt`

## Workflow Overview

GCIG operates in two main phases:

### Phase 1: Enhanced Knowledge Graph Construction

```
Text Corpus → Entity Extraction → Summarization → Entity Alignment → Graph Connection
    ↓               ↓                   ↓                ↓                  ↓
Raw Text    entities/relations   +embeddings      +alias mapping     text_units
```

**Output:** Structured knowledge graph with entity-relation triplets and linked text units

### Phase 2: Cross-document Instruction Generation

```
Knowledge Graph → Chunk Selection → Question Gen → Relevance Check → Filtering → Answer Gen
      ↓                 ↓                 ↓              ↓              ↓           ↓
   EKG Data      Multi-hop Chains    Diverse QA    Coherence Score   Top-K    GraphRAG
```

**Output:** High-quality question-answer pairs with reasoning chains

**Key Innovations:**
- **LCS (LLM-driven Chunk Selection)**: Forms multi-hop evidence chains from knowledge graphs
- **APQG (Adaptive Prompt-based Question Generation)**: Generates 6 diverse question types
- **CoTAG (Chain-of-Thought-guided Answer Generation)**: Produces interpretable answers via GraphRAG reasoning

## Quick Start

### 1. Build Enhanced Knowledge Graph

#### Configure Environment

First, configure your LLM settings in `EKG_construct/config.py`:

```python
# Data Paths
data_dir = "/path/to/input/corpus"      # Input corpus directory
db_dir = "/path/to/output"               # Output directory

# LLM Configuration
class LLMConfig:
    model = "gpt-4"                      # Model identifier
    api_key = "your-api-key"             # API authentication
    base_url = "https://api.openai.com/v1"
```

#### Run EKG Construction Pipeline

Execute the complete 4-stage pipeline:

```bash
cd EKG_construct

# Run all stages sequentially
bash construct.sh

# Or run stages individually:
# Stage 1: Entity extraction
python graph_mult_construct.py

# Stage 2: Summarization & embedding
python graph_summary.py

# Stage 3: Entity alignment
python graph_align.py

# Stage 4: Text unit connection
python graph_connect.py
```

**Output Files:**
- `entities.parquet` - Entity nodes with descriptions and embeddings
- `relations.parquet` - Relation edges with descriptions and embeddings
- `text_units-{expand}-{threshold}.parquet` - Linked text units

### 2. Generate Cross-document Instructions

#### Configure Environment

Set up your API credentials:

```bash
# Create .env file
cat > instruct_generate/.env << EOF
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_MODEL=gpt-4
EOF
```

#### Run Instruction Generation Pipeline

```bash
cd instruct_generate

# Stage 1: Extract entity-based contexts (LCS)
python extract_entity.py --mode 2 --max_combinations 2000

# Stage 2: Generate diverse questions (APQG)
python question.py

# Stage 3: Assess cross-document relevance (optional)
python relevance.py

# Stage 4: Filter by quality and confidence
python filtered.py --relevance yes --keep 3000

# Stage 5: Generate answers with GraphRAG reasoning (CoTAG)
python answer.py
```

**Output:**
- `questions_with_answers.json` - Final QA pairs with reasoning chains
- `answer_N.json` - Checkpoint files at intervals
- `gpt_interaction_log.txt` - Detailed processing logs

### 3. Evaluate Generated Data

```bash
# Run evaluation on benchmark datasets
bash run_eval_all_models.sh
```

## Data Formats

### EKG Construction Outputs

**entities.parquet:**
```python
{
  "id": int,              # Unique entity identifier
  "name": str,            # Canonical entity name
  "type": str,            # Entity type (PERSON/ORGANIZATION/PRODUCT/LEGALTERM/CONDITION)
  "description": str,     # LLM-summarized description
  "embedding": array,     # Semantic embedding vector
  "alias": int/None       # Aligned entity ID (if duplicate)
}
```

**relations.parquet:**
```python
{
  "id": int,              # Unique relation identifier
  "source": int,          # Source entity ID
  "target": int,          # Target entity ID
  "description": str,     # Relation description
  "embedding": array      # Semantic embedding vector
}
```

**text_units-{expand}-{threshold}.parquet:**
```python
{
  "content": str,         # Text unit content
  "entities": list[int],  # Linked entity IDs
  "embedding": array      # Text embedding vector
}
```

### Instruction Generation Outputs

**questions_with_answers.json:**
```json
{
  "question_id": "string",
  "question": "string",
  "question_type": "Y/N | MCQ | EQA | NLI | SA | TC | MT |",
  "sources": ["text_unit_1", "text_unit_2", ...],
  "entities": ["entity_name_1", "entity_name_2", ...],
  "answer": "string",
  "reasoning_chain": ["step_1", "step_2", ...],
  "confidence": 0.0-1.0
}
```

**Question Types:**
- **Y/N**: Yes-or-No questions (binary factual verification)
- **MCQ**: Multiple-Choice Questions (selection from options)
- **EQA**: Extractive QA (span extraction from context)
- **NLI**: Natural Language Inference (logical relationship classification)
- **SA**: Sentiment Analysis (opinion polarity detection)
- **TC**: Topic Classification (thematic categorization)
- **MT**: Multi-hop QA (Multi-hop Question Answering)

## Experimental Results

<p align="center">
  <img src="resources/4.png" width="90%" alt="GCIG Framework Overview"/>
</p>

## Components

### EKG Construction Module (`EKG_construct/`)

Implements a 4-stage pipeline for automated knowledge graph construction:

**Stage 1: Entity Extraction** (`graph_mult_construct.py`)
- Multi-threaded parallel processing (15 threads)
- LLM-based entity and relation extraction
- Supports entity types: PERSON, ORGANIZATION, PRODUCT, LEGALTERM, CONDITION
- Output: `entities.parquet`, `relations.parquet`

**Stage 2: Summarization** (`graph_summary.py`)
- Consolidates multiple entity/relation descriptions using LLM
- Generates semantic embeddings for retrieval
- Batch processing for efficiency

**Stage 3: Entity Alignment** (`graph_align.py`)
- Multi-strategy alignment: type-based, similarity-based (cosine > 0.9), LLM-based
- Resolves entity ambiguity and merges duplicates
- Outputs alias mappings for entity disambiguation

**Stage 4: Graph Connection** (`graph_connect.py`)
- Links text units to knowledge graph entities
- Sentence-level entity recognition with context expansion
- Duplicate filtering with configurable threshold

See [EKG_construct/README.MD](EKG_construct/README.MD) for detailed specifications.

### Instruction Generation Module (`instruct_generate/`)

Implements GCIG's three core components across a 5-stage pipeline:

**Stage 1: LLM-driven Chunk Selection (LCS)** (`extract_entity.py`)
- Multi-hop subgraph retrieval with configurable depth
- BM25-based text unit ranking
- Combinatorial context generation (1-4 text units)
- Forms multi-hop evidence chains with factual consistency

**Stage 2: Adaptive Prompt-based Question Generation (APQG)** (`question.py`)
- Generates 6 question types: Y/N, MCQ, EQA, NLI, SA, TC
- Asynchronous batch processing (8 concurrent calls)
- Context-aware generation with entity/relation information

**Stage 3: Relevance Assessment** (`relevance.py`)
- Evaluates cross-document coherence and thematic overlap
- Binary classification with confidence metrics
- Expert content analyst prompting

**Stage 4: Quality Filtering** (`filtered.py`)
- Confidence-based scoring and ranking
- Top-K selection (default: 3000)
- Relevance filtering support

**Stage 5: Chain-of-Thought-guided Answer Generation (CoTAG)** (`answer.py`)
- GraphRAG-enhanced multi-hop reasoning
- Iterative relation exploration with entity pruning
- LLM/BM25/SentenceBERT-based candidate ranking
- Generates reasoning chains with answers

See [instruct_generate/README.md](instruct_generate/README.md) for detailed specifications.

### GraphGen

A framework for synthetic data generation guided by knowledge graphs. Supports:
- Multiple LLM inference backends (OpenAI, Ollama, vLLM, SGLang)
- Various data sources (files, databases, search engines)
- Multiple output formats (Atomic, Aggregated, CoT, Multi-hop, VQA)

See [GraphGen README](GraphGen/README.md) for detailed documentation.

### Baselines

We compare against several baseline methods:
- Self-Instruct
- Bonito
- RAG-Instruct
- EntiGraph

## Configuration

### EKG Construction Configuration

Edit `EKG_construct/config.py`:

```python
# Data Paths
data_dir = "/path/to/input"              # Input corpus directory
db_dir = "/path/to/output"               # Output directory

# LLM Configuration
class LLMConfig:
    model = "gpt-4"                      # Model identifier
    api_key = "your-api-key"             # API authentication
    base_url = "https://api.openai.com/v1"
    temperature = 0.0                    # Generation temperature
    max_tokens = 4096                    # Maximum output tokens

# Processing Parameters
class SplitterConfig:
    chunk_size = 2048                    # Token-based chunking
    over_lap = 64                        # Overlap between chunks
    encoding_name = "cl100k_base"        # Tokenizer encoding

class GraphExtractorConfig:
    max_gleanings = 1                    # LLM refinement iterations
    num_threads = 15                     # Parallel processing threads

class ConnectorConfig:
    expand = 1                           # Context expansion window (sentences)
    overlap_threshold = 0.9              # Duplicate detection threshold

class SimilarityAlignConfig:
    threshold = 0.9                      # Entity alignment threshold
```

### Instruction Generation Configuration

Create `.env` file in `instruct_generate/`:

```bash
# API Configuration
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Model Configuration
DEFAULT_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
```

Edit module-specific parameters in respective scripts:

**extract_entity.py:**
```python
mode = 2                    # Text units per combination (1-4)
max_combinations = 2000     # Maximum combinations to generate
top_k = 3-5                 # Text units to retrieve per entity
```

**question.py:**
```python
MAX_CONCURRENT = 8          # Concurrent LLM calls
BATCH_SIZE = 100            # Files per processing batch
```

**answer.py:**
```python
USE_GRAPH_REASONING = True  # Enable GraphRAG reasoning
REASONING_DEPTH = 3         # Maximum exploration depth
REASONING_WIDTH = 3         # Candidates per reasoning step
MAX_CONCURRENT = 8          # Concurrent threads
BATCH_SIZE = 80             # Questions per batch
```

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
