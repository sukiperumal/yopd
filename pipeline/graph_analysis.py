#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import json
import networkx as nx
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """Analyze multimodal brain graphs"""
    
    def __init__(self):
        self.graphs = {}
        self.summaries = {}
        self.analysis_results = {}
    
    def load_subject_graphs(self, graph_dir: Path, subject_pattern: str = "*") -> int:
        """Load all subject graph files from directory"""
        try:
            graph_files = list(graph_dir.glob(f"sub-{subject_pattern}_multimodal_graph.pkl"))
            
            for graph_file in graph_files:
                subject_id = graph_file.stem.replace("_multimodal_graph", "").replace("sub-", "")
                
                with open(graph_file, 'rb') as f:
                    graph_data = pickle.load(f)
                
                self.graphs[subject_id] = graph_data
                logger.info(f"Loaded graph for subject {subject_id}")
            
            logger.info(f"Loaded {len(self.graphs)} subject graphs")
            return len(self.graphs)
            
        except Exception as e:
            logger.error(f"Failed to load graphs: {e}")
            raise
    
    def compute_network_metrics(self) -> pd.DataFrame:
        """Compute network-level metrics for all subjects"""
        try:
            metrics_list = []
            
            for subject_id, graph_data in self.graphs.items():
                multimodal_graph = graph_data['multimodal']['networkx_graph']
                functional_adj = graph_data['functional']['adjacency_matrix']
                
                # Basic graph properties
                n_nodes = multimodal_graph.number_of_nodes()
                n_edges = multimodal_graph.number_of_edges()
                density = nx.density(multimodal_graph)
                
                # Connectivity metrics
                try:
                    # Global efficiency
                    global_efficiency = nx.global_efficiency(multimodal_graph)
                    
                    # Average clustering
                    avg_clustering = nx.average_clustering(multimodal_graph)
                    
                    # Average path length (for largest connected component)
                    if nx.is_connected(multimodal_graph):
                        avg_path_length = nx.average_shortest_path_length(multimodal_graph)
                    else:
                        # Use largest connected component
                        largest_cc = max(nx.connected_components(multimodal_graph), key=len)
                        subgraph = multimodal_graph.subgraph(largest_cc)
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                    
                    # Small-world properties
                    small_world_sigma = avg_clustering / avg_path_length if avg_path_length > 0 else 0
                    
                    # Modularity (using Louvain algorithm)
                    try:
                        communities = nx.community.louvain_communities(multimodal_graph)
                        modularity = nx.community.modularity(multimodal_graph, communities)
                    except:
                        modularity = np.nan
                    
                    # Rich club coefficient (simplified)
                    degrees = dict(multimodal_graph.degree())
                    mean_degree = np.mean(list(degrees.values()))
                    
                except Exception as e:
                    logger.warning(f"Some metrics failed for {subject_id}: {e}")
                    global_efficiency = np.nan
                    avg_clustering = np.nan
                    avg_path_length = np.nan
                    small_world_sigma = np.nan
                    modularity = np.nan
                    mean_degree = np.nan
                
                # Functional connectivity metrics
                func_strength = np.mean(functional_adj[functional_adj > 0]) if np.any(functional_adj > 0) else 0
                func_sparsity = np.sum(functional_adj > 0) / (functional_adj.shape[0] * (functional_adj.shape[0] - 1))
                
                metrics = {
                    'subject_id': subject_id,
                    'n_nodes': n_nodes,
                    'n_edges': n_edges,
                    'density': density,
                    'global_efficiency': global_efficiency,
                    'avg_clustering': avg_clustering,
                    'avg_path_length': avg_path_length,
                    'small_world_sigma': small_world_sigma,
                    'modularity': modularity,
                    'mean_degree': mean_degree,
                    'functional_strength': func_strength,
                    'functional_sparsity': func_sparsity
                }
                
                metrics_list.append(metrics)
            
            df_metrics = pd.DataFrame(metrics_list)
            self.analysis_results['network_metrics'] = df_metrics
            
            logger.info(f"Computed network metrics for {len(df_metrics)} subjects")
            return df_metrics
            
        except Exception as e:
            logger.error(f"Failed to compute network metrics: {e}")
            raise
    
    def identify_hubs(self, percentile: float = 90) -> Dict[str, List]:
        """Identify hub regions based on multiple centrality measures"""
        try:
            hub_analysis = {}
            
            for subject_id, graph_data in self.graphs.items():
                multimodal_graph = graph_data['multimodal']['networkx_graph']
                roi_labels = graph_data['multimodal']['roi_labels']
                
                # Compute centrality measures
                degree_centrality = nx.degree_centrality(multimodal_graph)
                betweenness_centrality = nx.betweenness_centrality(multimodal_graph)
                closeness_centrality = nx.closeness_centrality(multimodal_graph)
                eigenvector_centrality = nx.eigenvector_centrality(multimodal_graph, max_iter=1000)
                
                # Convert to arrays
                n_nodes = len(roi_labels)
                centrality_scores = np.zeros((n_nodes, 4))
                
                for i in range(n_nodes):
                    centrality_scores[i, 0] = degree_centrality.get(i, 0)
                    centrality_scores[i, 1] = betweenness_centrality.get(i, 0)
                    centrality_scores[i, 2] = closeness_centrality.get(i, 0)
                    centrality_scores[i, 3] = eigenvector_centrality.get(i, 0)
                
                # Z-score normalize
                centrality_scores_norm = stats.zscore(centrality_scores, axis=0)
                
                # Composite hub score (average of normalized centralities)
                hub_scores = np.mean(centrality_scores_norm, axis=1)
                
                # Identify hubs (top percentile)
                hub_threshold = np.percentile(hub_scores, percentile)
                hub_indices = np.where(hub_scores >= hub_threshold)[0]
                
                hubs = [roi_labels[i] for i in hub_indices]
                
                hub_analysis[subject_id] = {
                    'hubs': hubs,
                    'hub_indices': hub_indices.tolist(),
                    'hub_scores': hub_scores[hub_indices].tolist(),
                    'centrality_measures': {
                        'degree': centrality_scores[:, 0].tolist(),
                        'betweenness': centrality_scores[:, 1].tolist(),
                        'closeness': centrality_scores[:, 2].tolist(),
                        'eigenvector': centrality_scores[:, 3].tolist()
                    }
                }
            
            self.analysis_results['hub_analysis'] = hub_analysis
            logger.info(f"Identified hubs for {len(hub_analysis)} subjects")
            
            return hub_analysis
            
        except Exception as e:
            logger.error(f"Failed to identify hubs: {e}")
            raise
    
    def analyze_multimodal_features(self) -> Dict:
        """Analyze the relationship between functional and structural features"""
        try:
            feature_analysis = {}
            
            # Collect all features across subjects
            all_functional_features = []
            all_structural_features = []
            all_multimodal_features = []
            subject_ids = []
            
            for subject_id, graph_data in self.graphs.items():
                multimodal = graph_data['multimodal']
                functional = graph_data['functional']
                structural = graph_data['structural']
                
                # Extract feature matrices
                multimodal_features = multimodal['multimodal_features']
                structural_features = structural['roi_features']
                
                # For functional features, use graph metrics
                func_metrics = functional['graph_metrics']
                functional_features = np.column_stack([
                    func_metrics[metric] for metric in func_metrics.keys()
                ])
                
                all_functional_features.append(functional_features)
                all_structural_features.append(structural_features)
                all_multimodal_features.append(multimodal_features)
                subject_ids.append(subject_id)
            
            # Concatenate across subjects
            func_features_concat = np.vstack(all_functional_features)
            struct_features_concat = np.vstack(all_structural_features)
            multimodal_features_concat = np.vstack(all_multimodal_features)
            
            # Dimensionality reduction
            # PCA
            pca = PCA(n_components=10)
            multimodal_pca = pca.fit_transform(multimodal_features_concat)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            multimodal_tsne = tsne.fit_transform(multimodal_features_concat[:1000])  # Sample for speed
            
            # Feature correlation analysis
            func_struct_corr = np.corrcoef(
                func_features_concat.T, 
                struct_features_concat.T
            )
            
            feature_analysis = {
                'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
                'n_components_95_variance': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95)) + 1,
                'func_struct_correlation': func_struct_corr.tolist(),
                'feature_names': {
                    'functional': list(graph_data['functional']['graph_metrics'].keys()),
                    'structural': graph_data['structural']['feature_names'],
                    'multimodal': graph_data['multimodal']['feature_names']
                }
            }
            
            self.analysis_results['feature_analysis'] = feature_analysis
            logger.info("Completed multimodal feature analysis")
            
            return feature_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze features: {e}")
            raise
    
    def create_group_consensus_network(self, threshold: float = 0.5) -> nx.Graph:
        """Create group-level consensus network"""
        try:
            n_subjects = len(self.graphs)
            
            # Get network size from first subject
            first_subject = list(self.graphs.values())[0]
            n_nodes = first_subject['multimodal']['n_rois']
            roi_labels = first_subject['multimodal']['roi_labels']
            
            # Sum adjacency matrices across subjects
            adjacency_sum = np.zeros((n_nodes, n_nodes))
            
            for graph_data in self.graphs.values():
                adj_matrix = graph_data['multimodal']['multimodal_adjacency']
                # Binarize (1 if edge exists, 0 otherwise)
                binary_adj = (adj_matrix > 0).astype(int)
                adjacency_sum += binary_adj
            
            # Create consensus network (edges present in at least threshold proportion of subjects)
            consensus_threshold = int(n_subjects * threshold)
            consensus_adj = (adjacency_sum >= consensus_threshold).astype(int)
            
            # Create NetworkX graph
            consensus_graph = nx.from_numpy_array(consensus_adj)
            
            # Add node labels
            for i, label in enumerate(roi_labels):
                consensus_graph.nodes[i]['label'] = label
                consensus_graph.nodes[i]['consistency'] = adjacency_sum[i, :].sum() / n_subjects
            
            self.analysis_results['consensus_network'] = {
                'graph': consensus_graph,
                'adjacency_matrix': consensus_adj,
                'consistency_matrix': adjacency_sum / n_subjects,
                'threshold': threshold,
                'n_subjects': n_subjects
            }
            
            logger.info(f"Created consensus network with {consensus_graph.number_of_edges()} edges")
            return consensus_graph
            
        except Exception as e:
            logger.error(f"Failed to create consensus network: {e}")
            raise
    
    def save_analysis_results(self, output_dir: Path):
        """Save all analysis results"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save network metrics as CSV
            if 'network_metrics' in self.analysis_results:
                metrics_file = output_dir / "network_metrics.csv"
                self.analysis_results['network_metrics'].to_csv(metrics_file, index=False)
                logger.info(f"Saved network metrics to {metrics_file}")
            
            # Save hub analysis as JSON
            if 'hub_analysis' in self.analysis_results:
                hub_file = output_dir / "hub_analysis.json"
                with open(hub_file, 'w') as f:
                    json.dump(self.analysis_results['hub_analysis'], f, indent=2)
                logger.info(f"Saved hub analysis to {hub_file}")
            
            # Save feature analysis as JSON
            if 'feature_analysis' in self.analysis_results:
                feature_file = output_dir / "feature_analysis.json"
                with open(feature_file, 'w') as f:
                    json.dump(self.analysis_results['feature_analysis'], f, indent=2)
                logger.info(f"Saved feature analysis to {feature_file}")
            
            # Save consensus network
            if 'consensus_network' in self.analysis_results:
                consensus_file = output_dir / "consensus_network.pkl"
                with open(consensus_file, 'wb') as f:
                    pickle.dump(self.analysis_results['consensus_network'], f)
                logger.info(f"Saved consensus network to {consensus_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis results: {e}")
            raise

class GraphVisualizer:
    """Create visualizations for brain graph analysis"""
    
    def __init__(self, analyzer: GraphAnalyzer):
        self.analyzer = analyzer
        self.results = analyzer.analysis_results
    
    def plot_network_metrics_distribution(self, output_dir: Path):
        """Plot distribution of network metrics across subjects"""
        try:
            if 'network_metrics' not in self.results:
                logger.warning("Network metrics not available for plotting")
                return
            
            df = self.results['network_metrics']
            
            # Select numeric columns for plotting
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'subject_id']
            
            # Create subplots
            n_cols = 3
            n_rows = int(np.ceil(len(numeric_cols) / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=15, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'{col}')
                    axes[i].set_xlabel('Value')
                    axes[i].set_ylabel('Frequency')
            
            # Remove empty subplots
            for i in range(len(numeric_cols), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            
            output_file = output_dir / "network_metrics_distribution.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved network metrics distribution plot to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot network metrics: {e}")
    
    def plot_hub_consistency(self, output_dir: Path):
        """Plot hub consistency across subjects"""
        try:
            if 'hub_analysis' not in self.results:
                logger.warning("Hub analysis not available for plotting")
                return
            
            hub_analysis = self.results['hub_analysis']
            
            # Count hub frequency across subjects
            all_hubs = []
            for subject_data in hub_analysis.values():
                all_hubs.extend(subject_data['hubs'])
            
            hub_counts = pd.Series(all_hubs).value_counts()
            
            # Plot top 20 most consistent hubs
            top_hubs = hub_counts.head(20)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(top_hubs)), top_hubs.values)
            plt.yticks(range(len(top_hubs)), [label.split('_')[-1] for label in top_hubs.index])
            plt.xlabel('Number of Subjects')
            plt.title('Most Consistent Hub Regions Across Subjects')
            plt.gca().invert_yaxis()
            
            output_file = output_dir / "hub_consistency.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved hub consistency plot to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot hub consistency: {e}")
    
    def plot_consensus_network(self, output_dir: Path):
        """Plot the consensus network"""
        try:
            if 'consensus_network' not in self.results:
                logger.warning("Consensus network not available for plotting")
                return
            
            consensus_data = self.results['consensus_network']
            G = consensus_data['graph']
            
            plt.figure(figsize=(12, 10))
            
            # Use spring layout for visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=0.5)
            
            # Draw nodes
            node_sizes = [G.degree(node) * 20 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                                 node_color='lightblue', alpha=0.8)
            
            plt.title(f'Consensus Network (threshold={consensus_data["threshold"]})')
            plt.axis('off')
            
            output_file = output_dir / "consensus_network.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved consensus network plot to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to plot consensus network: {e}")
    
    def create_all_plots(self, output_dir: Path):
        """Create all available plots"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_network_metrics_distribution(output_dir)
        self.plot_hub_consistency(output_dir)
        self.plot_consensus_network(output_dir)
        
        logger.info(f"All plots saved to {output_dir}")

def run_complete_analysis(graph_dir: Path, output_dir: Path, subject_pattern: str = "*"):
    """Run complete graph analysis pipeline"""
    try:
        # Initialize analyzer
        analyzer = GraphAnalyzer()
        
        # Load graphs
        n_subjects = analyzer.load_subject_graphs(graph_dir, subject_pattern)
        
        if n_subjects == 0:
            logger.error("No graphs loaded. Exiting.")
            return
        
        # Run analyses
        logger.info("Computing network metrics...")
        analyzer.compute_network_metrics()
        
        logger.info("Identifying hub regions...")
        analyzer.identify_hubs()
        
        logger.info("Analyzing multimodal features...")
        analyzer.analyze_multimodal_features()
        
        logger.info("Creating consensus network...")
        analyzer.create_group_consensus_network()
        
        # Save results
        logger.info("Saving analysis results...")
        analyzer.save_analysis_results(output_dir)
        
        # Create visualizations
        logger.info("Creating visualizations...")
        visualizer = GraphVisualizer(analyzer)
        visualizer.create_all_plots(output_dir / "plots")
        
        logger.info("Complete analysis finished successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise