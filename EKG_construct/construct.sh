#!/bin/bash

################################################################################
# Entity Knowledge Graph Construction Pipeline
#
# Description: Sequential execution of the four-stage EKG construction pipeline
#              for automated knowledge graph generation from unstructured text.
#
# Stages:
#   1. Entity and Relation Extraction (graph_mult_construct.py)
#   2. Semantic Summarization and Embedding (graph_summary.py)
#   3. Entity Alignment and Disambiguation (graph_align.py)
#   4. Text Unit Connection and Contextualization (graph_connect.py)
#
# Usage: bash construct.sh
################################################################################

set -e  # Exit immediately if any command fails
set -u  # Treat unset variables as errors
set -o pipefail  # Pipe failures cause script to exit

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Error handler
error_exit() {
    log_error "Pipeline execution failed at stage: $1"
    log_error "Please check the logs above for detailed error information."
    exit 1
}

# Record start time
PIPELINE_START=$(date +%s)

echo "================================================================================"
log_info "Starting Entity Knowledge Graph Construction Pipeline"
echo "================================================================================"
echo ""

# Stage 1: Entity and Relation Extraction
echo "--------------------------------------------------------------------------------"
log_info "Stage 1/4: Entity and Relation Extraction"
log_info "Executing: graph_mult_construct.py"
echo "--------------------------------------------------------------------------------"
STAGE_START=$(date +%s)

python graph_mult_construct.py || error_exit "Entity Extraction (Stage 1)"

STAGE_END=$(date +%s)
STAGE_DURATION=$((STAGE_END - STAGE_START))
log_success "Stage 1 completed in ${STAGE_DURATION}s"
echo ""

# Stage 2: Semantic Summarization and Embedding
echo "--------------------------------------------------------------------------------"
log_info "Stage 2/4: Semantic Summarization and Embedding"
log_info "Executing: graph_summary.py"
echo "--------------------------------------------------------------------------------"
STAGE_START=$(date +%s)

python graph_summary.py || error_exit "Summarization and Embedding (Stage 2)"

STAGE_END=$(date +%s)
STAGE_DURATION=$((STAGE_END - STAGE_START))
log_success "Stage 2 completed in ${STAGE_DURATION}s"
echo ""

# Stage 3: Entity Alignment and Disambiguation
echo "--------------------------------------------------------------------------------"
log_info "Stage 3/4: Entity Alignment and Disambiguation"
log_info "Executing: graph_align.py"
echo "--------------------------------------------------------------------------------"
STAGE_START=$(date +%s)

python graph_align.py || error_exit "Entity Alignment (Stage 3)"

STAGE_END=$(date +%s)
STAGE_DURATION=$((STAGE_END - STAGE_START))
log_success "Stage 3 completed in ${STAGE_DURATION}s"
echo ""

# Stage 4: Text Unit Connection
echo "--------------------------------------------------------------------------------"
log_info "Stage 4/4: Text Unit Connection and Contextualization"
log_info "Executing: graph_connect.py"
echo "--------------------------------------------------------------------------------"
STAGE_START=$(date +%s)

python graph_connect.py || error_exit "Graph Connection (Stage 4)"

STAGE_END=$(date +%s)
STAGE_DURATION=$((STAGE_END - STAGE_START))
log_success "Stage 4 completed in ${STAGE_DURATION}s"
echo ""

# Calculate total execution time
PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

# Final summary
echo "================================================================================"
log_success "Entity Knowledge Graph Construction Pipeline Completed Successfully"
echo "================================================================================"
log_info "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
log_info "Output directory: Check config.py for db_dir location"
echo "================================================================================"

exit 0