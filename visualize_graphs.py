#!/usr/bin/env python3
"""
Graph Visualization Script

Visualize constructed brain graphs with:
1. Connectivity matrices (heatmaps)
2. 3D brain network plots
3. Node feature distributions
4. Graph statistics

Author: YOPD Pipeline
Date: November 2025
"""

import sys
import argparse
import pickle
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import networkx as nx
import pandas as pd

# Neuroimaging visualization
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualize brain graphs"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def load_graph(self, graph_file: Path) -> Dict:
        """Load graph from pickle file"""
        with open(graph_file, 'rb') as f:
            return pickle.load(f)
    
    def visualize_connectivity_matrix(self, graph_data: Dict, output_file: Path):
        """Create connectivity matrix heatmap"""
        logger.info(f"Creating connectivity matrix visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Similarity matrix
        ax = axes[0]
        im1 = ax.imshow(graph_data['similarity_matrix'], cmap='RdBu_r', 
                       vmin=-1, vmax=1, aspect='auto')
        ax.set_title(f"Morphometric Similarity Matrix\n{graph_data['subject_id']}", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Brain Region')
        ax.set_ylabel('Brain Region')
        plt.colorbar(im1, ax=ax, label='Correlation')
        
        # Adjacency matrix
        ax = axes[1]
        im2 = ax.imshow(graph_data['adjacency_matrix'], cmap='binary', aspect='auto')
        ax.set_title(f"Binary Adjacency Matrix\n{graph_data['graph_metrics']['num_edges']} edges", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Brain Region')
        ax.set_ylabel('Brain Region')
        plt.colorbar(im2, ax=ax, label='Connection (0/1)')
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_file.name}")
    
    def visualize_node_features(self, graph_data: Dict, output_file: Path):
        """Visualize node feature distributions"""
        logger.info(f"Creating node feature visualization...")
        
        features = graph_data['raw_node_features']
        feature_names = graph_data['feature_names']
        
        n_features = len(feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, (name, ax) in enumerate(zip(feature_names, axes)):
            if idx < features.shape[1]:
                ax.hist(features[:, idx], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_title(f'{name}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = np.mean(features[:, idx])
                std_val = np.std(features[:, idx])
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                ax.legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Node Feature Distributions - {graph_data["subject_id"]}', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_file.name}")
    
    def visualize_centrality_measures(self, graph_data: Dict, output_file: Path):
        """Visualize node centrality measures"""
        logger.info(f"Creating centrality visualization...")
        
        metrics = graph_data['node_metrics']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Degree centrality
        ax = axes[0]
        ax.bar(range(len(metrics['degree_centrality'])), 
               sorted(metrics['degree_centrality'], reverse=True),
               color='steelblue', alpha=0.7)
        ax.set_title('Degree Centrality', fontsize=12, fontweight='bold')
        ax.set_xlabel('ROI (sorted)')
        ax.set_ylabel('Centrality')
        ax.grid(True, alpha=0.3)
        
        # Betweenness centrality
        ax = axes[1]
        ax.bar(range(len(metrics['betweenness_centrality'])), 
               sorted(metrics['betweenness_centrality'], reverse=True),
               color='coral', alpha=0.7)
        ax.set_title('Betweenness Centrality', fontsize=12, fontweight='bold')
        ax.set_xlabel('ROI (sorted)')
        ax.set_ylabel('Centrality')
        ax.grid(True, alpha=0.3)
        
        # Closeness centrality
        ax = axes[2]
        ax.bar(range(len(metrics['closeness_centrality'])), 
               sorted(metrics['closeness_centrality'], reverse=True),
               color='mediumseagreen', alpha=0.7)
        ax.set_title('Closeness Centrality', fontsize=12, fontweight='bold')
        ax.set_xlabel('ROI (sorted)')
        ax.set_ylabel('Centrality')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Node Centrality Measures - {graph_data["subject_id"]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_file.name}")
    
    def visualize_graph_statistics(self, graph_data: Dict, output_file: Path):
        """Create comprehensive graph statistics summary"""
        logger.info(f"Creating graph statistics visualization...")
        
        metrics = graph_data['graph_metrics']
        subject_id = graph_data['subject_id']
        
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Graph metrics text
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        stats_text = f"""
        Subject: {subject_id}
        Atlas: {graph_data['atlas']}
        Number of Nodes: {metrics['num_nodes']}
        Number of Edges: {metrics['num_edges']}
        Network Density: {metrics['density']:.4f}
        Average Clustering: {metrics['avg_clustering']:.4f}
        """
        
        ax1.text(0.5, 0.5, stats_text, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.set_title('Graph Statistics Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Degree distribution
        ax2 = fig.add_subplot(gs[1, 0])
        G = nx.from_numpy_array(graph_data['adjacency_matrix'])
        degrees = [d for n, d in G.degree()]
        ax2.hist(degrees, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.set_title('Degree Distribution', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Edge weight distribution
        ax3 = fig.add_subplot(gs[1, 1])
        edge_weights = graph_data['edge_weights'][graph_data['adjacency_matrix'] == 1]
        ax3.hist(edge_weights, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax3.set_title('Edge Weight Distribution', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Correlation Strength')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Node feature correlation heatmap
        ax4 = fig.add_subplot(gs[2, :])
        feature_corr = np.corrcoef(graph_data['node_features'].T)
        im = ax4.imshow(feature_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax4.set_xticks(range(len(graph_data['feature_names'])))
        ax4.set_yticks(range(len(graph_data['feature_names'])))
        ax4.set_xticklabels(graph_data['feature_names'], rotation=45, ha='right')
        ax4.set_yticklabels(graph_data['feature_names'])
        ax4.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax4, label='Correlation')
        
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_file.name}")
    
    def create_group_summary(self, graph_files: List[Path], output_file: Path):
        """Create summary visualization across all subjects"""
        logger.info(f"Creating group summary visualization...")
        
        # Load all graphs and extract metrics
        subjects = []
        num_edges_list = []
        density_list = []
        clustering_list = []
        
        for graph_file in graph_files:
            try:
                graph_data = self.load_graph(graph_file)
                subjects.append(graph_data['subject_id'])
                num_edges_list.append(graph_data['graph_metrics']['num_edges'])
                density_list.append(graph_data['graph_metrics']['density'])
                clustering_list.append(graph_data['graph_metrics']['avg_clustering'])
            except Exception as e:
                logger.warning(f"Could not load {graph_file.name}: {e}")
                continue
        
        if not subjects:
            logger.warning("No graphs to summarize")
            return
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Number of edges distribution
        ax = axes[0, 0]
        ax.hist(num_edges_list, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(num_edges_list), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(num_edges_list):.0f}')
        ax.set_title('Distribution of Edge Counts', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Network density distribution
        ax = axes[0, 1]
        ax.hist(density_list, bins=20, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(density_list), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(density_list):.4f}')
        ax.set_title('Distribution of Network Density', fontsize=12, fontweight='bold')
        ax.set_xlabel('Density')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Clustering coefficient distribution
        ax = axes[1, 0]
        ax.hist(clustering_list, bins=20, color='mediumseagreen', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(clustering_list), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(clustering_list):.4f}')
        ax.set_title('Distribution of Average Clustering', fontsize=12, fontweight='bold')
        ax.set_xlabel('Clustering Coefficient')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_stats = f"""
        GROUP STATISTICS SUMMARY
        
        Total Subjects: {len(subjects)}
        
        Number of Edges:
          Mean: {np.mean(num_edges_list):.1f}
          Std: {np.std(num_edges_list):.1f}
          Range: [{np.min(num_edges_list)}, {np.max(num_edges_list)}]
        
        Network Density:
          Mean: {np.mean(density_list):.4f}
          Std: {np.std(density_list):.4f}
          Range: [{np.min(density_list):.4f}, {np.max(density_list):.4f}]
        
        Clustering Coefficient:
          Mean: {np.mean(clustering_list):.4f}
          Std: {np.std(clustering_list):.4f}
          Range: [{np.min(clustering_list):.4f}, {np.max(clustering_list):.4f}]
        """
        
        ax.text(0.1, 0.5, summary_stats, fontsize=10, ha='left', va='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        plt.suptitle('Group-Level Graph Statistics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize constructed brain graphs'
    )
    
    parser.add_argument(
        '--graph-dir',
        type=str,
        required=True,
        help='Directory containing graph pickle files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/sukiperumal/Documents/yopd/outputs/visualizations',
        help='Output directory for visualizations'
    )
    
    parser.add_argument(
        '--subjects',
        type=str,
        nargs='*',
        help='Specific subjects to visualize (default: first 5 + group summary)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Visualize all subjects'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("BRAIN GRAPH VISUALIZATION")
    logger.info("=" * 80)
    logger.info(f"Graph directory: {graph_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Find graph files
    graph_files = sorted(list(graph_dir.glob("*_structural_graph.pkl")))
    
    if not graph_files:
        logger.error("No graph files found!")
        return 1
    
    logger.info(f"Found {len(graph_files)} graph files")
    
    # Initialize visualizer
    visualizer = GraphVisualizer(output_dir)
    
    # Determine which subjects to visualize
    if args.subjects:
        selected_files = [f for f in graph_files 
                         if any(s in f.stem for s in args.subjects)]
    elif args.all:
        selected_files = graph_files
    else:
        # Default: first 5 subjects
        selected_files = graph_files[:5]
    
    logger.info(f"Visualizing {len(selected_files)} individual subjects...")
    logger.info("")
    
    # Visualize individual subjects
    for graph_file in selected_files:
        try:
            subject_id = graph_file.stem.replace('_structural_graph', '')
            logger.info(f"Visualizing {subject_id}...")
            
            # Load graph
            graph_data = visualizer.load_graph(graph_file)
            
            # Create visualizations
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(exist_ok=True)
            
            visualizer.visualize_connectivity_matrix(
                graph_data,
                subject_output_dir / f'{subject_id}_connectivity_matrices.png'
            )
            
            visualizer.visualize_node_features(
                graph_data,
                subject_output_dir / f'{subject_id}_node_features.png'
            )
            
            visualizer.visualize_centrality_measures(
                graph_data,
                subject_output_dir / f'{subject_id}_centrality.png'
            )
            
            visualizer.visualize_graph_statistics(
                graph_data,
                subject_output_dir / f'{subject_id}_statistics.png'
            )
            
            logger.info(f"✓ Completed {subject_id}\n")
            
        except Exception as e:
            logger.error(f"Failed to visualize {graph_file.name}: {e}")
            continue
    
    # Create group summary
    logger.info("Creating group summary visualization...")
    visualizer.create_group_summary(
        graph_files,
        output_dir / 'group_summary.png'
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Visualizations saved to: {output_dir}")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
