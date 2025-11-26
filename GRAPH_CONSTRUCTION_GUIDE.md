# Brain Graph Construction and Visualization Guide

## Overview

This guide walks you through constructing structural brain graphs from your 75 preprocessed subjects and visualizing the results.

**Data Available**: T1w structural MRI images with brain masks (GM, WM, CSF)  
**Graphs to Construct**: Structural connectivity graphs based on morphometric similarity  
**Note**: fMRI data is not available in the preprocessed directory, so we focus on structural graphs only.

---

## Prerequisites

✅ **Already Installed**:
- Python 3.14.0
- Poetry environment with all dependencies
- 75 validated subjects in `/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/`

---

## Step-by-Step Instructions

### Step 1: Make Scripts Executable

```bash
cd /Users/sukiperumal/Documents/yopd

# Make scripts executable
chmod +x batch_construct_structural_graphs.py
chmod +x visualize_graphs.py
```

---

### Step 2: Test on a Single Subject First

Before processing all 75 subjects, test with one subject to ensure everything works:

```bash
# Test with one subject (sub-YLOPD100)
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --n-rois 200 \
    --threshold 0.25 \
    --n-jobs 1 \
    --subjects YLOPD100
```

**Expected Output**:
- Creates `outputs/sub-YLOPD100_structural_graph.pkl`
- Creates `outputs/sub-YLOPD100_structural_graph_summary.json`
- Processing log in `graph_construction.log`

**Time**: ~2-3 minutes per subject

---

### Step 3: Process All 75 Subjects

Once the test works, process all subjects:

#### Option A: Sequential Processing (Safer, Slower)
```bash
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --n-rois 200 \
    --threshold 0.25 \
    --n-jobs 1
```

**Time**: ~2-4 hours for 75 subjects

#### Option B: Parallel Processing (Faster, More Memory)
```bash
# Use 4 parallel jobs
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --n-rois 200 \
    --threshold 0.25 \
    --n-jobs 4
```

**Time**: ~30-60 minutes with 4 cores

**What This Does**:
1. Loads Schaefer 200 ROI atlas
2. For each subject:
   - Resamples atlas to T1w space
   - Extracts morphometric features per ROI:
     - T1w intensity (mean, std)
     - GM probability (mean, std, volume)
     - WM probability (mean, volume)
     - ROI size
   - Computes morphometric similarity between all ROI pairs
   - Creates adjacency matrix (edges where correlation > 0.25)
   - Calculates graph metrics (degree, betweenness, closeness centrality)
   - Saves comprehensive graph data as pickle file

**Output Files**:
- `outputs/sub-*_structural_graph.pkl` - Complete graph data
- `outputs/sub-*_structural_graph_summary.json` - Quick summary
- `outputs/batch_processing_summary.json` - Overall processing report

---

### Step 4: Visualize Individual Subjects

After graph construction, visualize a few subjects to inspect results:

```bash
# Visualize first 5 subjects (default)
poetry run python visualize_graphs.py \
    --graph-dir outputs \
    --output-dir outputs/visualizations
```

Or visualize specific subjects:

```bash
# Visualize specific subjects
poetry run python visualize_graphs.py \
    --graph-dir outputs \
    --output-dir outputs/visualizations \
    --subjects YLOPD100 YLOPDHC01 YLOPD105
```

Or visualize ALL subjects:

```bash
# Visualize all 75 subjects (warning: generates many files)
poetry run python visualize_graphs.py \
    --graph-dir outputs \
    --output-dir outputs/visualizations \
    --all
```

**Time**: ~30 seconds per subject

**Visualizations Created** (for each subject):
1. **Connectivity Matrices** (`*_connectivity_matrices.png`)
   - Morphometric similarity matrix (correlation heatmap)
   - Binary adjacency matrix (connection map)

2. **Node Features** (`*_node_features.png`)
   - Distribution of all 8 morphometric features across ROIs
   - Histograms with mean values

3. **Centrality Measures** (`*_centrality.png`)
   - Degree centrality (how connected each region is)
   - Betweenness centrality (bridge regions)
   - Closeness centrality (proximity to other regions)

4. **Graph Statistics** (`*_statistics.png`)
   - Overall metrics (nodes, edges, density, clustering)
   - Degree distribution
   - Edge weight distribution
   - Feature correlation matrix

5. **Group Summary** (`group_summary.png`)
   - Population-level statistics across all subjects
   - Distribution comparisons

---

### Step 5: Verify Results

Check the outputs:

```bash
# Count generated graph files
ls -1 outputs/*_structural_graph.pkl | wc -l
# Expected: 75

# View processing summary
cat outputs/batch_processing_summary.json

# Check visualization directory
ls outputs/visualizations/

# View group summary
open outputs/visualizations/group_summary.png
# (or use your image viewer)
```

---

### Step 6: Run GNN Pipeline with New Graphs

Once all graphs are constructed, run the complete GNN pipeline:

```bash
poetry run python run_complete_gnn_pipeline.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir gnn_pipeline/outputs \
    --graphs-dir /Users/sukiperumal/Documents/yopd/outputs \
    --skip-graph-building
```

**This will**:
- Validate BIDS structure (✓ Already done)
- Load all 75 structural graphs
- Train GNN models (GCN, GAT, Combined architectures)
- Perform cross-validation
- Generate explainability analysis
- Create comprehensive visualizations
- Save results to `gnn_pipeline/outputs/`

**Time**: ~30-60 minutes depending on hardware

---

## Parameters Explained

### Graph Construction Parameters

**`--n-rois`**: Number of brain regions (ROIs)
- Options: 100, 200, 400, 600, 800, 1000
- Default: 200
- More ROIs = finer parcellation but longer processing
- 200 is a good balance

**`--threshold`**: Correlation threshold for creating edges
- Range: 0.0 to 1.0
- Default: 0.25
- Lower = denser network (more edges)
- Higher = sparser network (fewer edges)
- 0.25 keeps moderate to strong connections

**`--n-jobs`**: Number of parallel processing jobs
- 1 = sequential (safer, uses less memory)
- 4-8 = parallel (faster, more memory)
- -1 = use all CPU cores

### Visualization Parameters

**`--subjects`**: Specific subjects to visualize
- Provide space-separated list
- Example: `--subjects YLOPD100 YLOPD105 YLOPDHC01`

**`--all`**: Visualize all subjects
- Generates many files (75 × 4 = 300 images)
- Useful for quality control

---

## Output File Structure

After completion, you'll have:

```
outputs/
├── sub-YLOPD100_structural_graph.pkl          # Graph data (75 files)
├── sub-YLOPD100_structural_graph_summary.json # Summaries (75 files)
├── batch_processing_summary.json              # Overall summary
└── visualizations/
    ├── sub-YLOPD100/                          # Per-subject folders
    │   ├── sub-YLOPD100_connectivity_matrices.png
    │   ├── sub-YLOPD100_node_features.png
    │   ├── sub-YLOPD100_centrality.png
    │   └── sub-YLOPD100_statistics.png
    ├── sub-YLOPD105/
    │   └── ...
    └── group_summary.png                      # Population statistics
```

---

## Troubleshooting

### Issue: "Atlas download failed"

**Solution**: The first run needs to download the Schaefer atlas (~100MB). Ensure internet connection.

```bash
# Manually download atlas (if needed)
poetry run python -c "from nilearn import datasets; datasets.fetch_atlas_schaefer_2018(n_rois=200)"
```

### Issue: "Out of memory" during parallel processing

**Solution**: Reduce number of parallel jobs

```bash
# Use fewer parallel jobs
--n-jobs 2
# Or sequential processing
--n-jobs 1
```

### Issue: Subject processing fails

**Solution**: Check the log file for specific errors

```bash
tail -50 graph_construction.log

# Try processing that specific subject alone to see detailed error
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --subjects YLOPD100
```

### Issue: Visualization fails

**Solution**: Ensure matplotlib backend is set correctly

```bash
# If running on a server without display
export MPLBACKEND=Agg

# Then retry visualization
poetry run python visualize_graphs.py --graph-dir outputs --output-dir outputs/visualizations
```

---

## Advanced Options

### Custom Atlas Size

For finer parcellation:

```bash
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs_400rois \
    --n-rois 400 \
    --threshold 0.3
```

### Batch Process Subgroups

Process patients and controls separately:

```bash
# Process only YOPD patients
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --subjects YLOPD*

# Process only healthy controls
poetry run python batch_construct_structural_graphs.py \
    --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" \
    --output-dir outputs \
    --subjects YLOPDHC*
```

### Export Graph Statistics to CSV

```bash
# Create a Python script to extract statistics
poetry run python -c "
import pickle
import pandas as pd
from pathlib import Path

data = []
for f in Path('outputs').glob('*_structural_graph_summary.json'):
    with open(f) as file:
        import json
        summary = json.load(file)
        data.append(summary)

df = pd.DataFrame(data)
df.to_csv('graph_statistics.csv', index=False)
print(f'Exported statistics for {len(df)} subjects')
"
```

---

## Expected Results

After successful completion, you should have:

1. **75 structural brain graphs** - One per subject
2. **Morphometric features** - 8 features per ROI (200 ROIs)
3. **Connectivity matrices** - 200×200 similarity and adjacency matrices
4. **Graph metrics** - Centrality measures for each node
5. **Visualizations** - Comprehensive plots for inspection
6. **Summary statistics** - Group-level comparison

These graphs can then be used for:
- GNN classification (distinguishing YOPD subtypes)
- Network analysis (identifying connectivity patterns)
- Biomarker discovery (finding discriminative features)
- Clinical correlation (relating structure to symptoms)

---

## Next Steps After Graph Construction

1. **Quality Control**: Review visualizations for anomalies
2. **Run GNN Pipeline**: Train classification models
3. **Statistical Analysis**: Compare patient vs control graphs
4. **Clinical Correlation**: Link graph metrics to clinical measures

---

## Quick Command Reference

```bash
# Single subject test
poetry run python batch_construct_structural_graphs.py --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" --output-dir outputs --subjects YLOPD100

# All subjects (sequential)
poetry run python batch_construct_structural_graphs.py --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" --output-dir outputs --n-jobs 1

# All subjects (parallel)
poetry run python batch_construct_structural_graphs.py --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" --output-dir outputs --n-jobs 4

# Visualize first 5
poetry run python visualize_graphs.py --graph-dir outputs --output-dir outputs/visualizations

# Visualize all
poetry run python visualize_graphs.py --graph-dir outputs --output-dir outputs/visualizations --all

# Run GNN pipeline
poetry run python run_complete_gnn_pipeline.py --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/" --output-dir gnn_pipeline/outputs --graphs-dir outputs --skip-graph-building
```

---

## Contact & Support

For issues or questions:
- Check `graph_construction.log` for detailed error messages
- Review `outputs/batch_processing_summary.json` for processing status
- Inspect individual subject visualizations for data quality

---

**Generated**: 2025-11-26  
**Pipeline Version**: 1.0.0
