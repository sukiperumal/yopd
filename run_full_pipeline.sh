#!/bin/bash
###############################################################################
# Complete Pipeline Execution Script
# 
# This script runs the entire pipeline:
# 1. Construct structural graphs for all subjects
# 2. Visualize graphs
# 3. Run GNN classification pipeline
#
# Usage: ./run_full_pipeline.sh [--test|--fast|--full]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/"
OUTPUT_DIR="outputs"
VIZ_DIR="outputs/visualizations"
GNN_OUTPUT_DIR="gnn_pipeline/outputs"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        YOPD Brain Graph Analysis Pipeline                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
MODE="${1:-test}"

case "$MODE" in
    --test)
        echo -e "${YELLOW}Running in TEST MODE (1 subject)${NC}"
        SUBJECTS="YLOPD100"
        N_JOBS=1
        VIZ_FLAG="--subjects YLOPD100"
        ;;
    --fast)
        echo -e "${YELLOW}Running in FAST MODE (parallel processing)${NC}"
        SUBJECTS=""
        N_JOBS=4
        VIZ_FLAG=""
        ;;
    --full)
        echo -e "${YELLOW}Running in FULL MODE (all subjects, all visualizations)${NC}"
        SUBJECTS=""
        N_JOBS=4
        VIZ_FLAG="--all"
        ;;
    *)
        echo -e "${YELLOW}Running in DEFAULT MODE (all subjects, sample visualizations)${NC}"
        SUBJECTS=""
        N_JOBS=1
        VIZ_FLAG=""
        ;;
esac

echo ""
echo -e "${GREEN}Step 1/3: Constructing Structural Brain Graphs${NC}"
echo "================================================================"

if [ -n "$SUBJECTS" ]; then
    poetry run python batch_construct_structural_graphs.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --n-rois 200 \
        --threshold 0.25 \
        --n-jobs $N_JOBS \
        --subjects $SUBJECTS
else
    poetry run python batch_construct_structural_graphs.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --n-rois 200 \
        --threshold 0.25 \
        --n-jobs $N_JOBS
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Graph construction completed successfully${NC}"
else
    echo -e "${RED}✗ Graph construction failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 2/3: Visualizing Brain Graphs${NC}"
echo "================================================================"

if [ "$MODE" = "--test" ]; then
    poetry run python visualize_graphs.py \
        --graph-dir "$OUTPUT_DIR" \
        --output-dir "$VIZ_DIR" \
        $VIZ_FLAG
else
    poetry run python visualize_graphs.py \
        --graph-dir "$OUTPUT_DIR" \
        --output-dir "$VIZ_DIR" \
        $VIZ_FLAG
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualization completed successfully${NC}"
else
    echo -e "${RED}✗ Visualization failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Step 3/3: Running GNN Classification Pipeline${NC}"
echo "================================================================"

poetry run python run_complete_gnn_pipeline.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$GNN_OUTPUT_DIR" \
    --graphs-dir "$OUTPUT_DIR" \
    --skip-graph-building \
    --skip-validation

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GNN pipeline completed successfully${NC}"
else
    echo -e "${RED}✗ GNN pipeline failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              PIPELINE EXECUTION COMPLETE                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Results Summary:${NC}"
echo "  Graph files:        $OUTPUT_DIR/*_structural_graph.pkl"
echo "  Visualizations:     $VIZ_DIR/"
echo "  GNN results:        $GNN_OUTPUT_DIR/results/"
echo ""
echo -e "${YELLOW}View results:${NC}"
echo "  Graph summary:      cat $OUTPUT_DIR/batch_processing_summary.json"
echo "  Group stats:        open $VIZ_DIR/group_summary.png"
echo "  GNN summary:        cat $GNN_OUTPUT_DIR/results/RESULTS_SUMMARY.md"
echo ""
