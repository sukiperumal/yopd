#!/usr/bin/env python3

"""
Demo Script: Graph Construction Output Examples

This script demonstrates the structure and content of the graphs produced by
graph_t1w.py and graph_fmri.py by creating example outputs and showing their structure.

Since the actual data processing requires fMRIPrep and FreeSurfer outputs which may not
be available, this script creates synthetic example graphs that match the exact structure
and format that would be produced by the real pipeline.
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List
import networkx as nx
from dataclasses import asdict
import logging

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our configuration classes
import sys
sys.path.append('/Users/sukiperumal/Documents/yopd/pipeline')

try:
    from graph_t1w import StructuralConfig
    from graph_fmri import FunctionalConfig
except ImportError as e:
    logger.warning(f"Could not import config classes: {e}")
    # Define minimal config classes
    class StructuralConfig:
        def __init__(self, **kwargs):
            self.schaefer_n_rois = kwargs.get('schaefer_n_rois', 200)
            self.morphometric_features = kwargs.get('morphometric_features', ['thickness', 'area', 'grayvol'])
            self.similarity_threshold = kwargs.get('similarity_threshold', 0.1)
            self.normalize_features = kwargs.get('normalize_features', True)
    
    class FunctionalConfig:
        def __init__(self, **kwargs):
            self.schaefer_n_rois = kwargs.get('schaefer_n_rois', 200)
            self.correlation_threshold = kwargs.get('correlation_threshold', 0.3)
            self.connectivity_metric = kwargs.get('connectivity_metric', 'correlation')

def create_example_roi_labels(n_rois: int) -> List[str]:
    """Create example Schaefer atlas ROI labels"""
    labels = []
    networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    
    rois_per_hemisphere = n_rois // 2
    
    for hemi in ['LH', 'RH']:
        for i in range(rois_per_hemisphere):
            network = networks[i % len(networks)]
            region_num = (i // len(networks)) + 1
            labels.append(f"7Networks_{hemi}_{network}_{region_num}")
    
    return labels

def create_structural_graph_example(n_rois: int = 200) -> Dict:
    """
    Create an example structural graph that matches the output format of graph_t1w.py
    """
    logger.info(f"Creating example structural graph with {n_rois} ROIs")
    
    # Configuration
    config = StructuralConfig(
        schaefer_n_rois=n_rois,
        morphometric_features=['thickness', 'area', 'grayvol', 'curvature'],
        similarity_threshold=0.1,
        normalize_features=True
    )
    
    # ROI labels (Schaefer atlas style)
    roi_labels = create_example_roi_labels(n_rois)
    
    # Morphometric features (n_rois × n_features)
    n_features = len(config.morphometric_features)
    roi_features = np.random.normal(0, 1, (n_rois, n_features))
    
    # Simulate realistic feature values
    roi_features[:, 0] = np.abs(np.random.normal(2.5, 0.3, n_rois))  # thickness (mm)
    roi_features[:, 1] = np.abs(np.random.normal(500, 100, n_rois))  # area (mm²)
    roi_features[:, 2] = np.abs(np.random.normal(1000, 200, n_rois))  # gray matter volume (mm³)
    roi_features[:, 3] = np.random.normal(0.1, 0.02, n_rois)  # mean curvature
    
    # Apply z-score normalization (as would be done in real pipeline)
    roi_features = (roi_features - np.mean(roi_features, axis=0)) / np.std(roi_features, axis=0)
    
    # Structural similarity matrix (morphometric similarity)
    similarity_matrix = np.corrcoef(roi_features)
    similarity_matrix = np.abs(similarity_matrix)  # Take absolute values
    np.fill_diagonal(similarity_matrix, 0)  # No self-connections
    
    # Apply threshold
    thresholded_similarity = np.where(
        similarity_matrix >= config.similarity_threshold,
        similarity_matrix,
        0
    )
    
    # Create NetworkX graph
    structural_network = nx.from_numpy_array(thresholded_similarity)
    
    # Add node attributes
    for i, (label, features) in enumerate(zip(roi_labels, roi_features)):
        structural_network.nodes[i]['label'] = label
        structural_network.nodes[i]['features'] = features.tolist()
        
        # Add individual feature attributes
        for j, feat_name in enumerate(config.morphometric_features):
            structural_network.nodes[i][feat_name] = float(features[j])
    
    # Add edge weights
    for u, v, data in structural_network.edges(data=True):
        data['weight'] = float(thresholded_similarity[u, v])
        data['similarity'] = float(thresholded_similarity[u, v])
    
    # Example cortical data (FreeSurfer aparc stats format)
    cortical_data = {
        'aparc': pd.DataFrame({
            'StructName': ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal'] * 2,
            'NumVert': np.random.randint(1000, 5000, 6),
            'SurfArea': np.random.randint(500, 2000, 6),
            'GrayVol': np.random.randint(1000, 4000, 6),
            'ThickAvg': np.random.uniform(2.0, 4.0, 6),
            'ThickStd': np.random.uniform(0.3, 0.8, 6),
            'MeanCurv': np.random.uniform(-0.2, 0.2, 6),
            'hemisphere': ['L'] * 3 + ['R'] * 3
        })
    }
    
    # Example subcortical data
    subcortical_data = pd.DataFrame({
        'StructName': ['Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala'],
        'Volume_mm3': [4000, 3900, 1600, 1550],
        'normMean': [45, 46, 48, 47],
        'normStdDev': [5, 4.8, 6.2, 5.9]
    })
    
    # Example brain volumes
    brain_volumes = {
        'BrainSegVol-to-eTIV': 0.82,
        'lhCortexVol': 250000,
        'rhCortexVol': 248000,
        'TotalGrayVol': 650000,
        'eTIV': 1500000
    }
    
    # Create the structural graph output dictionary
    structural_graph = {
        'roi_features': roi_features,
        'similarity_matrix': thresholded_similarity,
        'structural_network': structural_network,
        'feature_names': config.morphometric_features,
        'roi_labels': roi_labels,
        'n_rois': n_rois,
        'cortical_data': cortical_data,
        'subcortical_data': subcortical_data,
        'brain_volumes': brain_volumes,
        'config': config,
        'freesurfer_dir': '/example/path/to/freesurfer/sub-01'
    }
    
    logger.info(f"Created structural graph with {structural_network.number_of_nodes()} nodes and {structural_network.number_of_edges()} edges")
    
    return structural_graph

def create_functional_graph_example(n_rois: int = 200, n_timepoints: int = 300) -> Dict:
    """
    Create an example functional graph that matches the output format of graph_fmri.py
    """
    logger.info(f"Creating example functional graph with {n_rois} ROIs and {n_timepoints} timepoints")
    
    # Configuration
    config = FunctionalConfig(
        schaefer_n_rois=n_rois,
        correlation_threshold=0.3,
        connectivity_metric='correlation'
    )
    
    # ROI labels (same as structural)
    roi_labels = create_example_roi_labels(n_rois)
    
    # Simulate BOLD time series (n_timepoints × n_rois)
    # Create some realistic temporal structure
    time_series = np.random.normal(0, 1, (n_timepoints, n_rois))
    
    # Add some temporal autocorrelation and cross-regional correlations
    for t in range(1, n_timepoints):
        time_series[t] = 0.7 * time_series[t-1] + 0.3 * time_series[t]
    
    # Add network structure (regions in same network are more correlated)
    network_effects = np.random.normal(0, 0.5, (n_timepoints, 7))  # 7 networks
    for i, label in enumerate(roi_labels):
        for net_idx, net_name in enumerate(['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']):
            if net_name in label:
                time_series[:, i] += network_effects[:, net_idx]
    
    # Compute correlation matrix
    connectivity_matrix = np.corrcoef(time_series.T)
    
    # Apply threshold for adjacency matrix
    adjacency_matrix = np.where(
        connectivity_matrix > config.correlation_threshold,
        connectivity_matrix,
        0
    )
    np.fill_diagonal(adjacency_matrix, 0)
    
    # Compute graph metrics
    G = nx.from_numpy_array(np.abs(adjacency_matrix))
    
    # Graph-theoretic metrics
    graph_metrics = {
        'degree': np.array(list(dict(G.degree()).values())),
        'strength': np.array([sum([G[node][neighbor]['weight'] 
                                 for neighbor in G.neighbors(node)]) 
                            for node in G.nodes()]),
        'betweenness': np.array(list(nx.betweenness_centrality(G).values())),
        'closeness': np.array(list(nx.closeness_centrality(G).values())),
        'clustering': np.array(list(nx.clustering(G).values())),
        'local_efficiency': np.array([nx.local_efficiency(G, node) for node in G.nodes()]),
        'eigenvector': np.array(list(nx.eigenvector_centrality(G, max_iter=1000).values()))
    }
    
    # Create functional NetworkX graph with attributes
    functional_network = nx.from_numpy_array(adjacency_matrix)
    
    # Add node attributes
    for i, label in enumerate(roi_labels):
        functional_network.nodes[i]['label'] = label
        functional_network.nodes[i]['time_series_mean'] = float(np.mean(time_series[:, i]))
        functional_network.nodes[i]['time_series_std'] = float(np.std(time_series[:, i]))
        
        # Add graph metrics as node attributes
        for metric_name, metric_values in graph_metrics.items():
            functional_network.nodes[i][metric_name] = float(metric_values[i])
    
    # Create the functional graph output dictionary
    functional_graph = {
        'adjacency_matrix': adjacency_matrix,
        'connectivity_matrix': connectivity_matrix,
        'time_series': time_series,
        'graph_metrics': graph_metrics,
        'functional_network': functional_network,
        'roi_labels': roi_labels,
        'n_rois': n_rois,
        'threshold': config.correlation_threshold,
        'connectivity_metric': config.connectivity_metric,
        'tr': 2.0,  # Example TR in seconds
        'confounds_used': True,
        'n_timepoints': n_timepoints,
        'bold_file': '/example/path/to/bold.nii.gz',
        'config': config
    }
    
    logger.info(f"Created functional graph with {functional_network.number_of_nodes()} nodes and {functional_network.number_of_edges()} edges")
    
    return functional_graph

def save_graph_examples():
    """Create and save example graphs to outputs directory"""
    outputs_dir = Path('/Users/sukiperumal/Documents/yopd/outputs')
    outputs_dir.mkdir(exist_ok=True)
    
    logger.info("Creating example graph outputs...")
    
    # Create example graphs
    structural_graph = create_structural_graph_example(n_rois=200)
    functional_graph = create_functional_graph_example(n_rois=200, n_timepoints=300)
    
    # Save structural graph
    struct_file = outputs_dir / "example_structural_graph.pkl"
    with open(struct_file, 'wb') as f:
        pickle.dump(structural_graph, f)
    logger.info(f"Saved structural graph to {struct_file}")
    
    # Save functional graph  
    func_file = outputs_dir / "example_functional_graph.pkl"
    with open(func_file, 'wb') as f:
        pickle.dump(functional_graph, f)
    logger.info(f"Saved functional graph to {func_file}")
    
    # Create human-readable summaries
    create_graph_summaries(structural_graph, functional_graph, outputs_dir)
    
    return struct_file, func_file

def create_graph_summaries(structural_graph: Dict, functional_graph: Dict, output_dir: Path):
    """Create human-readable summaries of the graph structures"""
    
    # Structural graph summary
    struct_summary = {
        "graph_type": "structural",
        "description": "Brain graph derived from T1w MRI morphometric features",
        "data_structure": {
            "roi_features": {
                "type": "numpy.ndarray",
                "shape": list(structural_graph['roi_features'].shape),
                "description": "Morphometric features for each ROI (rows=ROIs, cols=features)",
                "features": structural_graph['feature_names']
            },
            "similarity_matrix": {
                "type": "numpy.ndarray", 
                "shape": list(structural_graph['similarity_matrix'].shape),
                "description": "Morphometric similarity matrix between ROIs",
                "range": [float(structural_graph['similarity_matrix'].min()), 
                         float(structural_graph['similarity_matrix'].max())],
                "n_edges": int(np.sum(structural_graph['similarity_matrix'] > 0) // 2),
                "sparsity": float(np.sum(structural_graph['similarity_matrix'] > 0) / 
                                (structural_graph['similarity_matrix'].shape[0] ** 2))
            },
            "structural_network": {
                "type": "networkx.Graph",
                "n_nodes": structural_graph['structural_network'].number_of_nodes(),
                "n_edges": structural_graph['structural_network'].number_of_edges(),
                "node_attributes": list(next(iter(structural_graph['structural_network'].nodes(data=True)))[1].keys()),
                "edge_attributes": list(next(iter(structural_graph['structural_network'].edges(data=True)))[2].keys())
            },
            "cortical_data": {
                "type": "dict of pandas.DataFrame",
                "description": "FreeSurfer cortical statistics",
                "parcellations": list(structural_graph['cortical_data'].keys()),
                "example_columns": list(structural_graph['cortical_data']['aparc'].columns)
            },
            "subcortical_data": {
                "type": "pandas.DataFrame", 
                "shape": list(structural_graph['subcortical_data'].shape),
                "description": "FreeSurfer subcortical volume statistics"
            },
            "brain_volumes": {
                "type": "dict",
                "description": "Global brain volume measures",
                "measures": list(structural_graph['brain_volumes'].keys())
            }
        },
        "roi_info": {
            "n_rois": structural_graph['n_rois'],
            "atlas": "Schaefer2018",
            "example_labels": structural_graph['roi_labels'][:5],
            "networks": list(set([label.split('_')[2] for label in structural_graph['roi_labels']]))
        },
        "processing_info": {
            "similarity_threshold": structural_graph['config'].similarity_threshold,
            "normalized_features": structural_graph['config'].normalize_features,
            "freesurfer_source": structural_graph['freesurfer_dir']
        }
    }
    
    # Functional graph summary
    func_summary = {
        "graph_type": "functional",
        "description": "Brain graph derived from fMRI BOLD functional connectivity",
        "data_structure": {
            "adjacency_matrix": {
                "type": "numpy.ndarray",
                "shape": list(functional_graph['adjacency_matrix'].shape),
                "description": "Thresholded functional connectivity matrix",
                "threshold": functional_graph['threshold'],
                "n_edges": int(np.sum(functional_graph['adjacency_matrix'] != 0) // 2),
                "sparsity": float(np.sum(functional_graph['adjacency_matrix'] != 0) / 
                                (functional_graph['adjacency_matrix'].shape[0] ** 2))
            },
            "connectivity_matrix": {
                "type": "numpy.ndarray",
                "shape": list(functional_graph['connectivity_matrix'].shape),
                "description": "Full correlation matrix (before thresholding)",
                "range": [float(functional_graph['connectivity_matrix'].min()),
                         float(functional_graph['connectivity_matrix'].max())]
            },
            "time_series": {
                "type": "numpy.ndarray",
                "shape": list(functional_graph['time_series'].shape),
                "description": "ROI BOLD time series (rows=timepoints, cols=ROIs)",
                "n_timepoints": functional_graph['n_timepoints'],
                "tr": functional_graph['tr']
            },
            "graph_metrics": {
                "type": "dict of numpy.ndarray",
                "description": "Graph-theoretic measures for each node",
                "metrics": list(functional_graph['graph_metrics'].keys()),
                "example_values": {
                    metric: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "range": [float(np.min(values)), float(np.max(values))]
                    }
                    for metric, values in functional_graph['graph_metrics'].items()
                }
            },
            "functional_network": {
                "type": "networkx.Graph",
                "n_nodes": functional_graph['functional_network'].number_of_nodes(),
                "n_edges": functional_graph['functional_network'].number_of_edges(),
                "node_attributes": list(next(iter(functional_graph['functional_network'].nodes(data=True)))[1].keys()),
                "edge_attributes": ["weight"] if functional_graph['functional_network'].edges() else []
            }
        },
        "roi_info": {
            "n_rois": functional_graph['n_rois'],
            "atlas": "Schaefer2018",
            "example_labels": functional_graph['roi_labels'][:5]
        },
        "acquisition_info": {
            "connectivity_metric": functional_graph['connectivity_metric'],
            "correlation_threshold": functional_graph['threshold'],
            "tr_seconds": functional_graph['tr'],
            "n_timepoints": functional_graph['n_timepoints'],
            "confounds_regression": functional_graph['confounds_used'],
            "bold_source": functional_graph['bold_file']
        }
    }
    
    # Save summaries
    struct_summary_file = output_dir / "structural_graph_structure.json"
    with open(struct_summary_file, 'w') as f:
        json.dump(struct_summary, f, indent=2)
    
    func_summary_file = output_dir / "functional_graph_structure.json"
    with open(func_summary_file, 'w') as f:
        json.dump(func_summary, f, indent=2)
    
    logger.info(f"Saved structural graph summary to {struct_summary_file}")
    logger.info(f"Saved functional graph summary to {func_summary_file}")
    
    # Create a comparison table
    comparison = {
        "graph_comparison": {
            "structural_graph": {
                "primary_data": "T1w MRI morphometry",
                "connectivity_basis": "Morphometric similarity",
                "node_features": structural_graph['feature_names'],
                "n_features_per_node": len(structural_graph['feature_names']),
                "n_edges": int(np.sum(structural_graph['similarity_matrix'] > 0) // 2),
                "edge_weight_meaning": "Morphometric similarity (0-1)",
                "temporal_dimension": False
            },
            "functional_graph": {
                "primary_data": "fMRI BOLD time series",
                "connectivity_basis": "Temporal correlation",
                "node_features": list(functional_graph['graph_metrics'].keys()),
                "n_features_per_node": len(functional_graph['graph_metrics']),
                "n_edges": int(np.sum(functional_graph['adjacency_matrix'] != 0) // 2),
                "edge_weight_meaning": "Correlation coefficient (-1 to 1)",
                "temporal_dimension": True,
                "temporal_length_minutes": functional_graph['n_timepoints'] * functional_graph['tr'] / 60
            },
            "shared_properties": {
                "atlas": "Schaefer2018",
                "n_rois": functional_graph['n_rois'],
                "roi_labels": "Identical across modalities",
                "graph_format": "NetworkX with node/edge attributes",
                "coordinate_space": "MNI152NLin2009cAsym"
            }
        }
    }
    
    comparison_file = output_dir / "graph_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Saved graph comparison to {comparison_file}")

def print_graph_structure_overview():
    """Print a formatted overview of the graph structures"""
    print("\n" + "="*80)
    print("MULTIMODAL BRAIN GRAPH PIPELINE - OUTPUT STRUCTURE OVERVIEW")
    print("="*80)
    
    print("""
📊 STRUCTURAL GRAPH (from graph_t1w.py)
────────────────────────────────────────
Input:  T1w MRI + FreeSurfer morphometry
Output: Dictionary with following structure:

{
  'roi_features': ndarray(n_rois, n_features)     # Morphometric features per ROI
  'similarity_matrix': ndarray(n_rois, n_rois)   # Morphometric similarity
  'structural_network': NetworkX.Graph            # Graph with node/edge attributes
  'feature_names': List[str]                      # ['thickness', 'area', 'volume', ...]
  'roi_labels': List[str]                         # Schaefer atlas region names
  'cortical_data': Dict[DataFrame]                # Raw FreeSurfer cortical stats
  'subcortical_data': DataFrame                   # Raw FreeSurfer subcortical volumes
  'brain_volumes': Dict                           # Global brain measures (ICV, etc.)
  'config': StructuralConfig                      # Processing parameters
}

📡 FUNCTIONAL GRAPH (from graph_fmri.py)  
──────────────────────────────────────────
Input:  Preprocessed fMRI BOLD + confounds
Output: Dictionary with following structure:

{
  'adjacency_matrix': ndarray(n_rois, n_rois)    # Thresholded connectivity
  'connectivity_matrix': ndarray(n_rois, n_rois) # Full correlation matrix
  'time_series': ndarray(n_timepoints, n_rois)   # ROI BOLD time series
  'graph_metrics': Dict[ndarray]                  # Node centrality measures
  'functional_network': NetworkX.Graph            # Graph with node/edge attributes
  'roi_labels': List[str]                         # Schaefer atlas region names
  'threshold': float                              # Correlation threshold used
  'tr': float                                     # Repetition time (seconds)
  'n_timepoints': int                             # Number of time points
  'config': FunctionalConfig                      # Processing parameters
}

🔗 MULTIMODAL INTEGRATION (from graph_merge.py)
─────────────────────────────────────────────────
Input:  Structural graph + Functional graph
Output: Dictionary with integrated representations

Both graphs use:
- Same ROI atlas (Schaefer 2018)
- Same coordinate space (MNI152NLin2009cAsym)  
- NetworkX format with rich node/edge attributes
- Comprehensive metadata and configuration info
""")

if __name__ == "__main__":
    print_graph_structure_overview()
    
    # Create example outputs
    struct_file, func_file = save_graph_examples()
    
    print(f"""
🎉 EXAMPLE OUTPUTS CREATED:
────────────────────────────
📁 Directory: /Users/sukiperumal/Documents/yopd/outputs/

📄 Files created:
  • example_structural_graph.pkl      - Complete structural graph data
  • example_functional_graph.pkl      - Complete functional graph data  
  • structural_graph_structure.json   - Human-readable structural summary
  • functional_graph_structure.json   - Human-readable functional summary
  • graph_comparison.json             - Side-by-side comparison

💡 To explore the graphs:
  
  import pickle
  import networkx as nx
  
  # Load structural graph
  with open('outputs/example_structural_graph.pkl', 'rb') as f:
      struct_graph = pickle.load(f)
  
  # Access components
  roi_features = struct_graph['roi_features']           # (200, 4) morphometric features
  similarity = struct_graph['similarity_matrix']       # (200, 200) similarity matrix
  nx_graph = struct_graph['structural_network']        # NetworkX graph
  
  # Load functional graph
  with open('outputs/example_functional_graph.pkl', 'rb') as f:
      func_graph = pickle.load(f)
      
  # Access components
  time_series = func_graph['time_series']               # (300, 200) BOLD time series
  connectivity = func_graph['connectivity_matrix']     # (200, 200) correlation matrix
  metrics = func_graph['graph_metrics']                # Graph measures per node

See the JSON files for detailed structure descriptions!
    """)