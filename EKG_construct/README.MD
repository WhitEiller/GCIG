# EKG Construction Module

## Overview

The **Entity Knowledge Graph (EKG) Construction Module** is a comprehensive pipeline for automated knowledge graph construction from unstructured textual data. This module leverages Large Language Models (LLMs) to extract entities and relations, perform semantic summarization, entity alignment, and contextual linking, ultimately generating a structured, queryable knowledge representation.

## System Architecture

The construction pipeline comprises four sequential stages, each implemented as an independent module:

```
Text Corpus → Entity Extraction → Summarization & Embedding → Entity Alignment → Graph Connection
```

### Pipeline Stages

1. **Multi-threaded Graph Construction** (`graph_mutl_construct.py`)
2. **Semantic Summarization and Embedding** (`graph_summary.py`)
3. **Entity Alignment and Disambiguation** (`graph_align.py`)
4. **Text Unit Connection** (`graph_connect.py`)

## Module Descriptions

### 1. Graph Construction (`graph_mutl_construct.py`)

**Purpose**: Extract structured entities and relations from unstructured text using LLM-based information extraction.

**Key Features**:
- Multi-threaded parallel processing for scalability
- Robust error handling with retry mechanisms
- Incremental graph building with real-time persistence
- Support for five entity types: `PERSON`, `ORGANIZATION`, `PRODUCT`, `LEGALTERM`, `CONDITION`

**Technical Details**:
- **Text Preprocessing**: Removes control characters, normalizes whitespace, handles text length constraints
- **Safe Extraction**: Implements retry logic with exponential backoff
- **Graph Merging**: Thread-safe merge operations for concurrent entity and relation updates
- **Output Format**: Parquet files (`entities.parquet`, `relations.parquet`)

**Parameters**:
- `max_gleanings`: Number of LLM refinement iterations (default: 1)
- `chunk_size`: Text segmentation size (configured via `SplitterConfig`)
- `num_threads`: Parallel processing threads (default: 15)

### 2. Semantic Summarization (`graph_summary.py`)

**Purpose**: Consolidate multiple descriptions per entity/relation into coherent summaries and generate semantic embeddings.

**Key Features**:
- Parallel processing of entities and relations
- LLM-based summarization for multi-description consolidation
- Batch embedding generation for semantic search capabilities

**Technical Details**:
- **Description Consolidation**:
  - Single description → Direct usage
  - Multiple descriptions → LLM-generated summary
  - Empty descriptions → Null value
- **Embedding Generation**: Utilizes `get_embedding()` for vector representation
- **Concurrency**: Dual-threaded architecture for simultaneous entity and relation processing

**Output**:
- Updated `entities.parquet` with `description` and `embedding` fields
- Updated `relations.parquet` with `description` and `embedding` fields

### 3. Entity Alignment (`graph_align.py`)

**Purpose**: Resolve entity ambiguity and merge duplicate entities through multi-strategy alignment.

**Alignment Strategies** (applied sequentially):
1. **Type-based Alignment**: Filters candidates by entity type
2. **Similarity-based Alignment**: Cosine similarity on description embeddings (threshold: 0.9)
3. **LLM-based Alignment**: Final disambiguation using language model reasoning

**Technical Details**:
- **Input**: Entity objects with `id`, `name`, `description`, `type`, `desc_embedding`
- **Output**: Alias mappings stored in `entities.parquet`
- **Configuration**: Adjustable similarity threshold via `SimilarityAlignConfig`

### 4. Graph Connection (`graph_connect.py`)

**Purpose**: Link original text units to extracted knowledge graph entities for contextual retrieval.

**Key Features**:
- Sentence-level entity recognition
- Configurable context expansion
- Overlap detection with threshold-based filtering
- Embedded text units for semantic search

**Technical Details**:
- **Context Expansion**: `expand` parameter controls neighboring sentence inclusion
- **Overlap Threshold**: Configurable similarity threshold for duplicate detection (default: 0.9)
- **Error Handling**: Graceful failure with skip mechanism for problematic texts
- **Output Naming**: `text_units-{expand}-{threshold}.parquet`

**Parameters**:
- `expand`: Context window size (sentences)
- `overlap_threshold`: Similarity cutoff for duplicate text units

## Configuration

All configurations are centralized in `config.py`:

### Data Paths
```python
data_dir = "/path/to/input"           # Input data directory
db_dir = "/path/to/output"            # Output directory
graph_dir = os.path.join(db_dir, "graph")
text_unit_dir = os.path.join(db_dir, "text_units")
```

### LLM Configuration
```python
class LLMConfig:
    model = "model-name"
    api_key = "your-api-key"
    base_url = "http://api-endpoint/v1"
```

### Processing Parameters
```python
class SplitterConfig:
    chunk_size = 2048
    over_lap = 64
    encoding_name = "cl100k_base"

class ConnectorConfig:
    expand = 1                    # Context expansion (sentences)
    overlap_threshold = 0.9       # Duplicate detection threshold

class SimilarityAlignConfig:
    threshold = 0.9               # Entity alignment threshold
```

## Execution

Execute the pipeline sequentially using the provided shell script:

```bash
bash construct.sh
```

**Pipeline Sequence**:
```bash
python graph_mult_construct.py    # Stage 1: Entity extraction
python graph_summary.py            # Stage 2: Summarization & embedding
python graph_align.py              # Stage 3: Entity alignment
python graph_connect.py            # Stage 4: Text unit connection
```

## Output Schema

### Entities (`entities.parquet`)
| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique entity identifier |
| `name` | str | Entity name |
| `type` | str | Entity type (PERSON/ORGANIZATION/etc.) |
| `description` | str | Summarized entity description |
| `embedding` | array | Semantic embedding vector |
| `alias` | int/None | Aligned entity ID (if duplicate) |

### Relations (`relations.parquet`)
| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique relation identifier |
| `source` | int | Source entity ID |
| `target` | int | Target entity ID |
| `description` | str/list | Relation description(s) |
| `embedding` | array | Semantic embedding vector |

### Text Units (`text_units-{expand}-{threshold}.parquet`)
| Field | Type | Description |
|-------|------|-------------|
| `content` | str | Text unit content |
| `entities` | list | Linked entity IDs |
| `embedding` | array | Text embedding vector |

## Dependencies

- `networkx`: Graph data structure
- `pandas`: DataFrame operations
- `pyarrow`: Parquet I/O
- `graphrag`: Custom graph extraction and alignment library
- `tqdm`: Progress visualization
- OpenAI-compatible LLM API

## Performance Considerations

- **Parallelization**: Multi-threaded processing with configurable thread pools
- **Incremental Persistence**: Real-time saving prevents data loss
- **Batch Embedding**: Reduces API calls through batch processing
- **Error Resilience**: Comprehensive exception handling with graceful degradation

## Citation

If you use this module in your research, please cite appropriately and acknowledge the GraphRAG framework.

## License

[Specify License]