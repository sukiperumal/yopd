# Complete Pipeline Execution Steps

## Quick Start (Choose One Method)

### Method 1: Automated Script (Recommended)

```bash
cd /Users/sukiperumal/Documents/yopd

# Test with one subject first (fast, ~3 minutes)
./run_full_pipeline.sh --test

# Run all subjects with parallel processing (recommended, ~1 hour)
./run_full_pipeline.sh --fast

# Run everything with all visualizations (slow, ~2-3 hours)
./run_full_pipeline.sh --full
```

### Method 2: Manual Step-by-Step Execution

Follow the detailed steps below for more control.

---

## Detailed Manual Steps

### STEP 1: Construct Brain Graphs (Required)

This creates structural connectivity graphs from T1w MRI data for all 75 subjects.

#### Option A: Test with One Subject First (Recommended)

```bash
cd /Users/sukiperumal/Documents/yopd

poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --n-rois 200 \
    --threshold 0.25 \
    --n-jobs 1 \
    --subjects YLOPD100
```

**What to expect:**
- Takes 2-3 minutes
- Creates `outputs/sub-YLOPD100_structural_graph.pkl`
- Creates `outputs/sub-YLOPD100_structural_graph_summary.json`
- Logs progress to console and `graph_construction.log`

**If successful, proceed to Option B. If errors occur, see troubleshooting section.**

#### Option B: Process All 75 Subjects

##### Sequential Processing (Safer, ~2-4 hours)
```bash
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --n-rois 200 \
    --threshold 0.25 \
    --n-jobs 1
```

##### Parallel Processing (Faster, ~30-60 minutes)
```bash
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --n-rois 200 \
    --threshold 0.25 \
    --n-jobs 4
```

**What this creates:**
- 75 graph files: `outputs/sub-*_structural_graph.pkl`
- 75 summary files: `outputs/sub-*_structural_graph_summary.json`
- Overall summary: `outputs/batch_processing_summary.json`

**Verify success:**
```bash
# Should show 75
ls -1 outputs/*_structural_graph.pkl | wc -l

# View summary
cat outputs/batch_processing_summary.json | grep -A 5 '"successful"'
```

---

### STEP 2: Visualize Graphs (Recommended)

Create visualizations to inspect graph quality and characteristics.

#### Option A: Quick Preview (First 5 Subjects)

```bash
poetry run python visualize_graphs.py \
    --graph-dir outputs \
    --output-dir outputs/visualizations
```

**Time:** ~2-3 minutes  
**Creates:** Visualizations for first 5 subjects + group summary

#### Option B: Specific Subjects

```bash
poetry run python visualize_graphs.py \
    --graph-dir outputs \
    --output-dir outputs/visualizations \
    --subjects YLOPD100 YLOPD105 YLOPDHC01 YLOPDHC02
```

#### Option C: All Subjects (Comprehensive)

```bash
poetry run python visualize_graphs.py \
    --graph-dir outputs \
    --output-dir outputs/visualizations \
    --all
```

**Time:** ~30-40 minutes  
**Creates:** 300+ visualization files (4 per subject × 75 subjects)

**What to inspect:**
- `outputs/visualizations/group_summary.png` - Population statistics
- `outputs/visualizations/sub-*/sub-*_connectivity_matrices.png` - Connection patterns
- `outputs/visualizations/sub-*/sub-*_centrality.png` - Important brain regions

**View a visualization:**
```bash
# Mac
open outputs/visualizations/group_summary.png

# Linux
xdg-open outputs/visualizations/group_summary.png

# Or use your preferred image viewer
```

---

### STEP 3: Run GNN Classification Pipeline (Main Analysis)

Train graph neural networks to classify subjects and identify connectivity patterns.

```bash
poetry run python run_complete_gnn_pipeline.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir gnn_pipeline/outputs \
    --graphs-dir outputs \
    --skip-graph-building \
    --skip-validation
```

**Time:** ~30-60 minutes depending on hardware

**What this does:**
1. Loads all 75 structural graphs
2. Prepares data with train/validation/test splits
3. Trains multiple GNN models:
   - BrainGCN (Graph Convolutional Network)
   - BrainGAT (Graph Attention Network)
   - BrainGCNGAT_Sequential (Combined model)
   - BrainGCNGAT_Parallel (Combined model)
4. Performs 5-fold cross-validation
5. Evaluates model performance
6. Generates explainability analysis
7. Creates comprehensive visualizations
8. Saves all results

**Output files:**
```
gnn_pipeline/outputs/
├── models/                          # Trained model checkpoints
├── results/
│   ├── RESULTS_SUMMARY.md          # Human-readable summary
│   ├── complete_pipeline_results.json
│   └── final_report.json
├── visualizations/                  # Performance plots
└── explainability/                  # Attention maps, ROI importance
```

**View results:**
```bash
# Summary report
cat gnn_pipeline/outputs/results/RESULTS_SUMMARY.md

# Detailed results
cat gnn_pipeline/outputs/results/complete_pipeline_results.json | python3 -m json.tool | less
```

---

## Verification Checklist

After completing all steps, verify:

### ✅ Graph Construction Complete

```bash
# Count graph files (should be 75)
ls -1 outputs/*_structural_graph.pkl | wc -l

# Check processing summary
cat outputs/batch_processing_summary.json

# Expected output should show:
# "successful": 75,
# "failed": 0
```

### ✅ Visualizations Created

```bash
# Check visualization directory
ls outputs/visualizations/

# Should contain:
# - group_summary.png
# - Multiple subject directories (sub-*)

# View group summary
open outputs/visualizations/group_summary.png
```

### ✅ GNN Pipeline Results

```bash
# Check results directory
ls gnn_pipeline/outputs/results/

# Should contain:
# - RESULTS_SUMMARY.md
# - complete_pipeline_results.json
# - final_report.json

# View summary
cat gnn_pipeline/outputs/results/RESULTS_SUMMARY.md
```

---

## Understanding the Output

### Graph Files (.pkl)

Each `*_structural_graph.pkl` contains:
- **n_rois**: 200 brain regions
- **node_features**: 8 morphometric features per region
- **similarity_matrix**: 200×200 correlation matrix
- **adjacency_matrix**: Binary connectivity (0/1)
- **graph_metrics**: Network properties (edges, density, clustering)
- **node_metrics**: Centrality measures per region

### Visualizations

1. **Connectivity Matrices**: Shows how brain regions are connected
2. **Node Features**: Distribution of morphometric measurements
3. **Centrality Measures**: Identifies hub regions
4. **Graph Statistics**: Overall network properties
5. **Group Summary**: Population-level comparisons

### GNN Results

1. **Model Performance**: Accuracy, F1-score, AUC for each model
2. **Cross-Validation**: Consistency across folds
3. **Feature Importance**: Which regions contribute to classification
4. **Attention Maps**: Where models focus (for GAT)
5. **Confusion Matrices**: Classification patterns

---

## Parameters Explained

### Graph Construction

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--n-rois` | 200 | 100, 200, 400, 600 | Number of brain regions |
| `--threshold` | 0.25 | 0.0-1.0 | Edge creation threshold |
| `--n-jobs` | 1 | 1-8, -1 | Parallel processing |

**Recommendations:**
- **n-rois**: 200 is optimal (balance between detail and processing)
- **threshold**: 0.25 for moderate connectivity (adjust 0.2-0.3 range)
- **n-jobs**: 4 for parallel, 1 for sequential (safer)

### GNN Training (in config)

Default settings in `gnn_pipeline/configs/default_config.json`:
- **epochs**: 200 (training iterations)
- **cv_folds**: 5 (cross-validation)
- **learning_rate**: 0.001
- **batch_size**: 16

---

## Troubleshooting

### Issue: "Atlas download failed"

```bash
# Manually download atlas
poetry run python -c "from nilearn import datasets; datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)"
```

### Issue: "Out of memory"

```bash
# Reduce parallel jobs
--n-jobs 1

# Or process in batches (patients first, then controls)
--subjects YLOPD*
```

### Issue: "Subject processing failed"

```bash
# Check log for details
tail -100 graph_construction.log

# Process problematic subject alone
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --subjects <SUBJECT_ID>
```

### Issue: "Visualization fails"

```bash
# Set matplotlib backend
export MPLBACKEND=Agg

# Then retry
poetry run python visualize_graphs.py --graph-dir outputs --output-dir outputs/visualizations
```

### Issue: "GNN pipeline error"

```bash
# Check if all graphs loaded
poetry run python -c "
from pathlib import Path
graphs = list(Path('outputs').glob('*_structural_graph.pkl'))
print(f'Found {len(graphs)} graph files')
"

# Should show 75
```

---

## Advanced Usage

### Process Specific Subgroups

```bash
# Only YOPD patients (50 subjects)
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs_patients \
    --subjects YLOPD100 YLOPD105 YLOPD106 # ... (list all)

# Only healthy controls (25 subjects)
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs_controls \
    --subjects YLOPDHC01 YLOPDHC02 # ... (list all)
```

### Different Atlas Resolutions

```bash
# Finer parcellation (400 ROIs)
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs_400rois \
    --n-rois 400 \
    --threshold 0.3
```

### Export Statistics to CSV

```bash
poetry run python -c "
import json
import pandas as pd
from pathlib import Path

summaries = []
for f in Path('outputs').glob('*_summary.json'):
    with open(f) as file:
        summaries.append(json.load(file))

df = pd.DataFrame(summaries)
df.to_csv('graph_statistics.csv', index=False)
print(f'Exported {len(df)} subjects to graph_statistics.csv')
"
```

---

## Expected Timeline

| Task | Sequential | Parallel (4 cores) |
|------|------------|-------------------|
| Test (1 subject) | 2-3 min | 2-3 min |
| All graphs (75) | 2-4 hours | 30-60 min |
| Visualize (5) | 2 min | 2 min |
| Visualize (all) | 30 min | 30 min |
| GNN pipeline | 30-60 min | 30-60 min |
| **Total (test)** | **5 min** | **5 min** |
| **Total (full)** | **3-5 hours** | **1.5-2 hours** |

---

## Summary of Commands

```bash
# Complete automated run (recommended)
./run_full_pipeline.sh --fast

# Or manual step-by-step:

# Step 1: Construct graphs
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs --n-rois 200 --threshold 0.25 --n-jobs 4

# Step 2: Visualize
poetry run python visualize_graphs.py \
    --graph-dir outputs --output-dir outputs/visualizations

# Step 3: Run GNN
poetry run python run_complete_gnn_pipeline.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir gnn_pipeline/outputs --graphs-dir outputs \
    --skip-graph-building --skip-validation

# Verify results
cat outputs/batch_processing_summary.json
open outputs/visualizations/group_summary.png
cat gnn_pipeline/outputs/results/RESULTS_SUMMARY.md
```

---

## What You'll Get

✅ **75 structural brain graphs** - Comprehensive connectivity networks  
✅ **Population statistics** - Group-level comparisons  
✅ **Individual visualizations** - Per-subject inspection  
✅ **GNN models** - Trained classification models  
✅ **Performance metrics** - Accuracy, precision, recall, F1  
✅ **Explainability** - Important regions and connections  
✅ **Research-ready outputs** - Publication-quality figures  

---

**Good luck with your analysis!**

For questions or issues, check:
- `graph_construction.log` - Graph construction details
- `outputs/batch_processing_summary.json` - Processing status
- `GRAPH_CONSTRUCTION_GUIDE.md` - Detailed technical guide

---

**Created**: 2025-11-26  
**Pipeline Version**: 1.0.0
