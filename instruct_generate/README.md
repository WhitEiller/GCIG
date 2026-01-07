# Instruction-based Question-Answer Generation Module

## Overview

The **Instruction-based Question-Answer Generation Module** is an automated pipeline for generating high-quality question-answer pairs from knowledge graphs and textual corpora. This module leverages knowledge graph retrieval, multi-hop reasoning, and Large Language Models (LLMs) to create diverse, contextually-grounded QA datasets suitable for training and evaluating retrieval-augmented generation systems.

## System Architecture

The generation pipeline comprises four sequential stages with optional quality control:

```
Knowledge Graph → Entity Extraction → Question Generation → Relevance Assessment → Filtering → Answer Generation
```

### Pipeline Stages

1. **Entity-based Context Extraction** (`extract_entity.py`)
2. **Question Generation** (`question.py`)
3. **Relevance Assessment** (`relevance.py`) [Optional]
4. **Quality Filtering** (`filtered.py`)
5. **Answer Generation with ToG/GraphRAG** (`answer.py`)

## Module Descriptions

### 1. Entity-based Context Extraction (`extract_entity.py`)

**Purpose**: Extract relevant subgraphs and text units from the knowledge graph based on entity-centric retrieval.

**Key Features**:
- Multi-hop subgraph retrieval with configurable depth
- BM25-based text unit ranking
- Combinatorial context generation (1-4 text units)
- Entity alias resolution and text unit linking

**Technical Details**:
- **Retrieval Strategy**:
  - Initial seed entity identification via BM25 similarity
  - Subgraph expansion using relation-based traversal
  - Text unit aggregation from entity neighborhoods
- **Combination Modes**: Supports 1-4 text unit combinations per entity
- **Deduplication**: Tracks processed combinations using frozenset hashing
- **Output Format**: Individual text files per combination in `question_outputs/`

**Parameters**:
- `mode`: Number of text units per combination (1-4)
- `max_combinations`: Maximum combinations to generate (default: 2000)
- `top_k`: Number of text units to retrieve (default: 3-5)

### 2. Question Generation (`question.py`)

**Purpose**: Generate diverse question types from extracted contexts using LLM-based prompting.

**Key Features**:
- Asynchronous batch processing with ThreadPoolExecutor
- Multiple question type support (Yes/No, Multiple Choice, Extractive, etc.)
- Context-aware question generation with entity and relation information
- Automatic answer appending to question files

**Technical Details**:
- **Question Types**:
  - Yes-or-No Question Answering
  - Multiple-Choice Question Answering
  - Extractive Question Answering
  - Natural Language Inference
  - Sentiment Analysis
  - Topic Classification
- **Processing**: Concurrent GPT calls with configurable parallelism (default: 8)
- **Error Handling**: Comprehensive logging with retry mechanisms
- **Output**: Appends `**Answer**` section to each question file

**Parameters**:
- `MAX_CONCURRENT`: Maximum concurrent LLM calls (default: 8)
- `BATCH_SIZE`: Files per batch (default: 100)
- `INPUT_DIR`: Directory containing question files

### 3. Relevance Assessment (`relevance.py`)

**Purpose**: Evaluate the logical relevance between multiple source texts using LLM-based reasoning.

**Key Features**:
- Automated extraction of source sentences from question files
- Expert content analyst prompting for relevance evaluation
- Yes/No relevance classification with confidence metrics
- Statistical reporting of relevance distribution

**Technical Details**:
- **Text Extraction**: Parses `**Sources**` section using AST literal evaluation
- **Evaluation Criteria**:
  - Thematic overlap between texts
  - Logical coherence and complementarity
  - Contextual relevance for question generation
- **Result Format**: Appends `**Relevance**` section with Yes/No classification
- **Post-processing**: Extracts and stores `**Relevance Result**` for filtering

**Parameters**:
- `MAX_CONCURRENT`: Parallel GPT calls (default: 4)
- `BATCH_SIZE`: Files per batch (default: 100)
- `RELEVANCE_MARKER`: Section delimiter (default: `**Relevance**`)

**Output**:
- Relevance assessment appended to each file
- Summary statistics (Yes/No/Unknown counts)
- Error logging for unparseable files

### 4. Quality Filtering (`filtered.py`)

**Purpose**: Filter and rank question-answer pairs based on confidence scores and relevance criteria.

**Key Features**:
- Multi-metric scoring (confidence sum, max confidence)
- Relevance-based filtering (Yes/No/Unknown)
- Top-K selection with configurable thresholds
- Automatic content cleaning (removes assessment metadata)

**Technical Details**:
- **Confidence Extraction**: Regex-based parsing of `Confidence (X): Y%` patterns
- **Scoring Strategy**:
  - Primary: Sum of all confidence scores
  - Secondary: Maximum individual confidence
  - Tertiary: Filename (alphabetical)
- **Filtering Modes**:
  - Threshold-based: Filter by minimum confidence percentage
  - Relevance-based: Filter by Yes/No relevance result
  - Hybrid: Combine both criteria
- **Cleaning**: Strips `**Relevance**` sections from output files

**Parameters**:
- `CONFIDENCE_THRESHOLD`: Minimum confidence percentage (default: 0)
- `KEEP_TOP_K`: Number of top items to retain (default: 3000)
- `--relevance`: Filter by Yes/No (default: 'yes')
- `--threshold`: Custom confidence threshold
- `--keep`: Custom top-K value

**Output**:
- Filtered files in `question_outputs_filtered/`
- Summary statistics with confidence ranges
- Relevance distribution in filtered set

### 5. Answer Generation with ToG/GraphRAG (`answer.py`)

**Purpose**: Generate high-quality answers using graph-based reasoning over knowledge graphs or direct LLM inference.

**Key Features**:
- **Dual Mode Operation**: ToG reasoning or direct GPT fallback
- **ToG Integration**: Multi-hop reasoning over knowledge graphs with Think-on-Graph algorithm
- **Asynchronous Processing**: Concurrent answer generation with progress tracking
- **Incremental Persistence**: Interval checkpointing to prevent data loss
- **Comprehensive Logging**: Detailed interaction logs with timing metrics

**Technical Details**:
- **ToG Reasoning Pipeline** (when enabled):
  1. Load knowledge graph (entities, relations, graph structure)
  2. Initialize ToG reasoner with graphrag's OpenAIModel
  3. Perform iterative graph exploration (configurable depth/width)
  4. Generate answers from reasoning chains
- **Graph Loading**:
  - Validates graph directory and parquet files existence
  - Gracefully falls back to direct GPT if graph unavailable
  - Supports custom graph paths via configuration
- **Processing Strategy**:
  - Skip already-answered questions (supports resume)
  - Batch processing with configurable size (default: 80)
  - Interval saving every N questions (default: 2000)
  - Rate limiting with sleep delays (10s between questions)
- **Output Schema**:
  - JSON format with question metadata, file sources, answer text
  - Detailed logging of prompts, responses, and processing times
  - Error logging for failed questions

**ToG/GraphRAG Configuration**:
```python
GRAPH_DIR = "/path/to/graph"  # Directory containing entities.parquet, relations.parquet
USE_TOG = True                 # Toggle ToG reasoning (True) or direct GPT (False)
TOG_DEPTH = 3                  # Maximum depth for graph exploration
TOG_WIDTH = 3                  # Number of top candidates per step
```

**Parameters**:
- `INPUT_JSON`: Input questions file (default: `extracted_questions_from_answer.json`)
- `OUTPUT_JSON`: Output QA pairs file (default: `questions_with_answers.json`)
- `MAX_CONCURRENT`: Concurrent processing threads (default: 8)
- `BATCH_SIZE`: Questions per batch (default: 80)
- `SAVE_INTERVAL`: Checkpoint frequency (default: 2000)

**Output**:
- `questions_with_answers.json`: Final QA pairs with metadata
- `answer_N.json`: Interval checkpoints at every N questions
- `gpt_interaction_log.txt`: Detailed processing log
- `error_log.txt`: Error tracking for debugging

