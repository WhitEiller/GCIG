# Instruction Generation Module

> Part of **GCIG: GraphRAG-based Cross-document Instruction Generation for Boosting LLM Reasoning**

## Overview

The **Instruction Generation Module** implements an automated pipeline for generating high-quality cross-document question-answer pairs from knowledge graphs and textual corpora. This module integrates three core components of the GCIG framework:

- **LLM-driven Chunk Selection (LCS)**: Forms multi-hop evidence chains with factual consistency
- **Adaptive Prompt-based Question Generation (APQG)**: Produces diverse multi-hop questions across multiple formats
- **Chain-of-Thought-guided Answer Generation (CoTAG)**: Generates interpretable answers with logical depth via GraphRAG reasoning

<p align="center">
  <img src="../resources/2.png" width="90%" alt="Instruction Generation Pipeline"/>
</p>

## System Architecture

The generation pipeline comprises five sequential stages with integrated quality control:

```
Knowledge Graph → Entity Extraction → Question Generation → Relevance Assessment → Filtering → Answer Generation
       ↓                 ↓                    ↓                     ↓                  ↓              ↓
    EKG Data      Multi-hop Chains      Diverse QA Types      Coherence Check     Top-K Select     GraphRAG
```

### Pipeline Stages

| Stage | Module | Component | Description |
|-------|--------|-----------|-------------|
| 1 | `extract_entity.py` | LCS | Entity-based context extraction with multi-hop retrieval |
| 2 | `question.py` | APQG | Adaptive prompt-based question generation |
| 3 | `relevance.py` | — | Cross-document relevance assessment |
| 4 | `filtered.py` | — | Quality filtering and ranking |
| 5 | `answer.py` | CoTAG | GraphRAG-enhanced answer generation |

## Module Specifications

### Stage 1: LLM-driven Chunk Selection (`extract_entity.py`)

**Purpose**: Extract relevant subgraphs and text units from the knowledge graph based on entity-centric retrieval, forming multi-hop evidence chains.

**Key Features**:
- Multi-hop subgraph retrieval with configurable depth
- BM25-based text unit ranking for relevance scoring
- Combinatorial context generation (1-4 text units per combination)
- Entity alias resolution and text unit linking

**Retrieval Strategy**:
```
Seed Entity (BM25) → Subgraph Expansion (Relations) → Text Unit Aggregation → Combination Generation
```

| Component | Method |
|-----------|--------|
| Initial Retrieval | BM25 similarity for seed entity identification |
| Subgraph Expansion | Relation-based graph traversal |
| Text Aggregation | Entity neighborhood text unit collection |
| Deduplication | Frozenset hashing for combination tracking |

**Configuration Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `mode` | Text units per combination | 1-4 |
| `max_combinations` | Maximum combinations to generate | 2000 |
| `top_k` | Text units to retrieve per entity | 3-5 |

**Output**: Individual text files per combination in `question_outputs/`

---

### Stage 2: Adaptive Prompt-based Question Generation (`question.py`)

**Purpose**: Generate diverse question types from extracted contexts using LLM-based prompting with adaptive templates.

**Supported Question Types**:

| Type | Abbreviation | Description |
|------|--------------|-------------|
| Yes-or-No QA | Y/N | Binary factual verification |
| Multiple-Choice QA | MCQ | Selection from predefined options |
| Extractive QA | EQA | Span extraction from context |
| Natural Language Inference | NLI | Logical relationship classification |
| Sentiment Analysis | SA | Opinion polarity detection |
| Topic Classification | TC | Thematic categorization |

**Technical Implementation**:
- Asynchronous batch processing via `ThreadPoolExecutor`
- Context-aware generation incorporating entity and relation information
- Automatic answer appending with confidence scoring

**Configuration Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_CONCURRENT` | Concurrent LLM calls | 8 |
| `BATCH_SIZE` | Files per processing batch | 100 |
| `INPUT_DIR` | Question file directory | `question_outputs/` |

---

### Stage 3: Relevance Assessment (`relevance.py`)

**Purpose**: Evaluate logical relevance between multiple source texts to ensure cross-document coherence.

**Evaluation Criteria**:
| Criterion | Description |
|-----------|-------------|
| Thematic Overlap | Shared topical elements between texts |
| Logical Coherence | Complementary information structure |
| Contextual Relevance | Suitability for question generation |

**Technical Implementation**:
- Expert content analyst prompting for relevance evaluation
- Binary classification (Yes/No) with confidence metrics
- Statistical reporting of relevance distribution

**Output Format**:
- `**Relevance**` section appended to each file
- `**Relevance Result**` extracted for downstream filtering
- Summary statistics (Yes/No/Unknown distribution)

**Configuration Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_CONCURRENT` | Parallel LLM calls | 4 |
| `BATCH_SIZE` | Files per batch | 100 |
| `RELEVANCE_MARKER` | Section delimiter | `**Relevance**` |

---

### Stage 4: Quality Filtering (`filtered.py`)

**Purpose**: Filter and rank question-answer pairs based on confidence scores and relevance criteria to ensure dataset quality.

**Scoring Strategy**:
| Priority | Metric | Description |
|----------|--------|-------------|
| Primary | Confidence Sum | Aggregate of all confidence scores |
| Secondary | Max Confidence | Highest individual confidence |
| Tertiary | Filename | Alphabetical ordering |

**Filtering Modes**:
| Mode | Description |
|------|-------------|
| Threshold-based | Minimum confidence percentage |
| Relevance-based | Yes/No relevance filtering |
| Hybrid | Combined criteria |

**Configuration Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `CONFIDENCE_THRESHOLD` | Minimum confidence | 0 |
| `KEEP_TOP_K` | Top items to retain | 3000 |
| `--relevance` | Relevance filter | 'yes' |

**Output**: Filtered files in `question_outputs_filtered/`

---

### Stage 5: Chain-of-Thought-guided Answer Generation (`answer.py`)

**Purpose**: Generate high-quality answers using graph-based reasoning over knowledge graphs, implementing the CoTAG component of GCIG.

**Dual Mode Operation**:

| Mode | Backend | Description |
|------|---------|-------------|
| Graph Reasoning | GraphRAG | Multi-hop graph exploration with reasoning chains |
| Direct Inference | LLM | Fallback when graph unavailable |

**GraphRAG Reasoning Pipeline**:
```
Question → Entity Linking → Iterative Relation Exploration → Entity Pruning → Answer Generation
              ↓                        ↓                          ↓                ↓
        Topic Entities         Depth-controlled           LLM/BM25/BERT      Reasoning Chain
                              Graph Traversal              Ranking
```

**GraphRAG Algorithm Components**:
| Component | Function |
|-----------|----------|
| Entity Linking | Topic entity identification from questions |
| Relation Exploration | Iterative relation discovery at configurable depth |
| Entity Pruning | LLM/BM25/SentenceBERT-based candidate filtering |
| Reasoning | Information sufficiency determination and answer generation |

**Configuration Parameters**:
```python
GRAPH_DIR = "/path/to/graph"  # EKG directory
USE_GRAPH_REASONING = True    # Enable graph-based reasoning
REASONING_DEPTH = 3           # Maximum exploration depth
REASONING_WIDTH = 3           # Candidates per step
```

**Processing Parameters**:
| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_CONCURRENT` | Concurrent threads | 8 |
| `BATCH_SIZE` | Questions per batch | 80 |
| `SAVE_INTERVAL` | Checkpoint frequency | 2000 |

**Output Schema**:
- `questions_with_answers.json`: Final QA pairs with metadata
- `answer_N.json`: Interval checkpoints
- `gpt_interaction_log.txt`: Detailed processing log
- `error_log.txt`: Error tracking

## Execution

### Full Pipeline Execution

```bash
# Stage 1: Extract entity-based contexts
python extract_entity.py --mode 2 --max_combinations 2000

# Stage 2: Generate questions
python question.py

# Stage 3: Assess relevance (optional)
python relevance.py

# Stage 4: Filter by quality
python filtered.py --relevance yes --keep 3000

# Stage 5: Generate answers with GraphRAG
python answer.py
```

### Configuration

Environment variables (`.env`):
```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_MODEL=gpt-4
```

## Output Schema

### Final QA Pairs (`questions_with_answers.json`)

```json
{
  "question_id": "string",
  "question": "string",
  "question_type": "string",
  "sources": ["string"],
  "entities": ["string"],
  "answer": "string",
  "reasoning_chain": ["string"],
  "confidence": "float"
}
```

## Dependencies

```
openai>=1.0           # LLM API client
pandas>=1.3           # Data manipulation
numpy>=1.21           # Numerical operations
tqdm>=4.62            # Progress visualization
rank_bm25>=0.2        # BM25 retrieval
sentence-transformers # Semantic similarity
graphrag              # Core GraphRAG framework
```

## Performance Considerations

| Aspect | Implementation |
|--------|----------------|
| Concurrency | Asynchronous batch processing with configurable parallelism |
| Persistence | Interval checkpointing prevents data loss |
| Rate Limiting | Configurable delays between API calls |
| Resilience | Comprehensive error handling with graceful degradation |
| Resume Support | Skip already-processed questions |

## Citation

If you use this module in your research, please cite:

```bibtex
@inproceedings{gcig2025,
  title={GCIG: GraphRAG-based Cross-document Instruction Generation for Boosting LLM Reasoning},
  author={},
  booktitle={},
  year={2025}
}
```

## License

This module is part of the GCIG project, licensed under the Apache License 2.0.
