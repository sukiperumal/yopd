#!/usr/bin/env python3

"""
Multimodal Graph Integration

This module integrates functional and structural brain graphs into unified
multimodal representations. It combines connectivity patterns and morphometric
features to create rich, multimodal brain networks.

Key components:
- Integrate functional and structural node features
- Combine connectivity matrices from multiple modalities
- Create unified multimodal brain graphs
- Compute multimodal network metrics

Author: Pipeline Development Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import pickle
from dataclasses import dataclass
import networkx as nx
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultimodalConfig:
    """Configuration for multimodal graph integration"""
    functional_weight: float = 0.6
    structural_weight: float = 0.4
    integration_method: str = 'weighted_average'  # 'weighted_average', 'concatenate', 'pca'
    normalize_before_integration: bool = True
    edge_integration_method: str = 'weighted_sum'  # 'weighted_sum', 'product', 'max'
    feature_selection_threshold: float = 0.95  # For PCA variance threshold
    create_separate_layers: bool = False  # Whether to maintain separate network layers
    
    def __post_init__(self):
        if self.functional_weight + self.structural_weight != 1.0:
            logger.warning("Functional and structural weights do not sum to 1.0")
            total = self.functional_weight + self.structural_weight
            self.functional_weight /= total
            self.structural_weight /= total

class MultimodalDataValidator:
    """Validate compatibility between functional and structural graph data"""
    
    @staticmethod
    def validate_graph_compatibility(functional_graph: Dict, structural_graph: Dict) -> bool:
        """Check if functional and structural graphs are compatible for integration"""
        try:
            # Check ROI counts
            func_n_rois = functional_graph.get('n_rois', 0)
            struct_n_rois = structural_graph.get('n_rois', 0)
            
            if func_n_rois != struct_n_rois:
                raise ValueError(f"ROI count mismatch: functional={func_n_rois}, structural={struct_n_rois}")
            
            # Check ROI labels
            func_labels = functional_graph.get('roi_labels', [])
            struct_labels = structural_graph.get('roi_labels', [])
            
            if len(func_labels) != len(struct_labels):
                raise ValueError(f"ROI label count mismatch: functional={len(func_labels)}, structural={len(struct_labels)}")
            
            # Check if labels match (order matters)
            if func_labels != struct_labels:
                logger.warning("ROI labels do not match exactly between modalities")
                # Could implement label matching logic here
            
            # Check matrix dimensions
            func_adj = functional_graph.get('adjacency_matrix')
            struct_sim = structural_graph.get('similarity_matrix')
            
            if func_adj is not None and struct_sim is not None:
                if func_adj.shape != struct_sim.shape:
                    raise ValueError(f"Adjacency matrix shape mismatch: {func_adj.shape} vs {struct_sim.shape}")
            
            logger.info("Graph compatibility validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Graph compatibility validation failed: {e}")
            raise

class NodeFeatureIntegrator:
    """Integrate node features from multiple modalities"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def extract_functional_features(self, functional_graph: Dict) -> np.ndarray:
        """Extract node features from functional graph"""
        try:
            graph_metrics = functional_graph.get('graph_metrics', {})
            time_series = functional_graph.get('time_series')
            
            feature_list = []
            feature_names = []
            
            # Graph-theoretic metrics
            for metric_name, metric_values in graph_metrics.items():
                if isinstance(metric_values, np.ndarray) and len(metric_values.shape) == 1:
                    feature_list.append(metric_values.reshape(-1, 1))
                    feature_names.append(f'func_{metric_name}')
            
            # Time series statistics if available
            if time_series is not None:
                # Mean and std of time series
                ts_mean = np.mean(time_series, axis=0)
                ts_std = np.std(time_series, axis=0)
                
                feature_list.extend([ts_mean.reshape(-1, 1), ts_std.reshape(-1, 1)])
                feature_names.extend(['func_ts_mean', 'func_ts_std'])
                
                # Variance
                ts_var = np.var(time_series, axis=0)
                feature_list.append(ts_var.reshape(-1, 1))
                feature_names.append('func_ts_variance')
            
            if feature_list:
                functional_features = np.hstack(feature_list)
                logger.info(f"Extracted functional features: {functional_features.shape}")
                return functional_features, feature_names
            else:
                raise ValueError("No functional features could be extracted")
                
        except Exception as e:
            logger.error(f"Failed to extract functional features: {e}")
            raise
    
    def extract_structural_features(self, structural_graph: Dict) -> Tuple[np.ndarray, List[str]]:
        """Extract node features from structural graph"""
        try:
            roi_features = structural_graph.get('roi_features')
            feature_names = structural_graph.get('feature_names', [])
            
            if roi_features is None:
                raise ValueError("No structural features found in graph data")
            
            # Add prefix to feature names
            prefixed_names = [f'struct_{name}' for name in feature_names]
            
            logger.info(f"Extracted structural features: {roi_features.shape}")
            return roi_features, prefixed_names
            
        except Exception as e:
            logger.error(f"Failed to extract structural features: {e}")
            raise
    
    def integrate_features(self, functional_features: np.ndarray, 
                          structural_features: np.ndarray,
                          func_feature_names: List[str],
                          struct_feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Integrate functional and structural features"""
        try:
            # Normalize features if requested
            if self.config.normalize_before_integration:
                functional_features = stats.zscore(functional_features, axis=0, nan_policy='omit')
                structural_features = stats.zscore(structural_features, axis=0, nan_policy='omit')
                
                # Handle NaN values
                functional_features = np.nan_to_num(functional_features, nan=0.0)
                structural_features = np.nan_to_num(structural_features, nan=0.0)
            
            if self.config.integration_method == 'concatenate':
                # Simple concatenation
                integrated_features = np.hstack([functional_features, structural_features])
                integrated_names = func_feature_names + struct_feature_names
                
            elif self.config.integration_method == 'weighted_average':
                # Weighted average (requires same number of features)
                min_features = min(functional_features.shape[1], structural_features.shape[1])
                
                func_subset = functional_features[:, :min_features]
                struct_subset = structural_features[:, :min_features]
                
                integrated_features = (self.config.functional_weight * func_subset + 
                                     self.config.structural_weight * struct_subset)
                
                integrated_names = [f'integrated_{i}' for i in range(min_features)]
                
            elif self.config.integration_method == 'pca':
                # PCA-based integration
                all_features = np.hstack([functional_features, structural_features])
                
                pca = PCA(n_components=self.config.feature_selection_threshold)
                integrated_features = pca.fit_transform(all_features)
                
                integrated_names = [f'pca_component_{i}' for i in range(integrated_features.shape[1])]
                
                logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
                
            else:
                # Default to concatenation
                integrated_features = np.hstack([functional_features, structural_features])
                integrated_names = func_feature_names + struct_feature_names
            
            logger.info(f"Integrated features: {integrated_features.shape}")
            return integrated_features, integrated_names
            
        except Exception as e:
            logger.error(f"Feature integration failed: {e}")
            raise

class EdgeIntegrator:
    """Integrate edge information from multiple modalities"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
    
    def integrate_adjacency_matrices(self, functional_adj: np.ndarray, 
                                   structural_adj: np.ndarray) -> np.ndarray:
        """Integrate functional and structural adjacency matrices"""
        try:
            # Ensure matrices have the same shape
            if functional_adj.shape != structural_adj.shape:
                raise ValueError(f"Matrix shape mismatch: {functional_adj.shape} vs {structural_adj.shape}")
            
            # Normalize matrices to [0, 1] range
            func_norm = self._normalize_matrix(functional_adj)
            struct_norm = self._normalize_matrix(structural_adj)
            
            if self.config.edge_integration_method == 'weighted_sum':
                # Weighted sum
                integrated_adj = (self.config.functional_weight * func_norm + 
                                self.config.structural_weight * struct_norm)
                
            elif self.config.edge_integration_method == 'product':
                # Element-wise product (both modalities must have connection)
                integrated_adj = func_norm * struct_norm
                
            elif self.config.edge_integration_method == 'max':
                # Element-wise maximum
                integrated_adj = np.maximum(func_norm, struct_norm)
                
            else:
                # Default to weighted sum
                integrated_adj = (self.config.functional_weight * func_norm + 
                                self.config.structural_weight * struct_norm)
            
            # Ensure diagonal is zero
            np.fill_diagonal(integrated_adj, 0)
            
            logger.info(f"Integrated adjacency matrices using {self.config.edge_integration_method}")
            return integrated_adj
            
        except Exception as e:
            logger.error(f"Edge integration failed: {e}")
            raise
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix to [0, 1] range"""
        try:
            # Handle different types of matrices
            matrix_abs = np.abs(matrix)
            
            matrix_min = matrix_abs.min()
            matrix_max = matrix_abs.max()
            
            if matrix_max == matrix_min:
                return matrix_abs  # All values are the same
            
            normalized = (matrix_abs - matrix_min) / (matrix_max - matrix_min)
            return normalized
            
        except Exception as e:
            logger.error(f"Matrix normalization failed: {e}")
            return matrix

class MultimodalNetworkCreator:
    """Create unified multimodal network representations"""
    
    def __init__(self, config: MultimodalConfig):
        self.config = config
    
    def create_multimodal_graph(self, integrated_adj: np.ndarray, 
                              integrated_features: np.ndarray,
                              feature_names: List[str],
                              roi_labels: List[str]) -> nx.Graph:
        """Create NetworkX graph with multimodal features"""
        try:
            # Create graph from integrated adjacency matrix
            G = nx.from_numpy_array(integrated_adj)
            
            # Add node attributes
            for i, (features, label) in enumerate(zip(integrated_features, roi_labels)):
                G.nodes[i]['label'] = label
                G.nodes[i]['features'] = features.tolist()
                
                # Add individual feature attributes
                for j, feat_name in enumerate(feature_names):
                    G.nodes[i][feat_name] = float(features[j])
            
            # Add edge weights
            for u, v, data in G.edges(data=True):
                data['weight'] = float(integrated_adj[u, v])
                data['multimodal_strength'] = float(integrated_adj[u, v])
            
            logger.info(f"Created multimodal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to create multimodal graph: {e}")
            raise
    
    def create_multilayer_graph(self, functional_graph: Dict, 
                              structural_graph: Dict) -> Dict:
        """Create multilayer network representation"""
        try:
            if not self.config.create_separate_layers:
                logger.info("Multilayer creation disabled in config")
                return {}
            
            # Extract adjacency matrices
            func_adj = functional_graph['adjacency_matrix']
            struct_adj = structural_graph['similarity_matrix']
            roi_labels = functional_graph['roi_labels']
            
            # Create separate layers
            functional_layer = nx.from_numpy_array(func_adj)
            structural_layer = nx.from_numpy_array(struct_adj)
            
            # Add layer-specific attributes
            for i, label in enumerate(roi_labels):
                # Functional layer
                functional_layer.nodes[i]['label'] = label
                functional_layer.nodes[i]['layer'] = 'functional'
                
                # Structural layer  
                structural_layer.nodes[i]['label'] = label
                structural_layer.nodes[i]['layer'] = 'structural'
            
            multilayer = {
                'functional_layer': functional_layer,
                'structural_layer': structural_layer,
                'interlayer_coupling': self._compute_interlayer_coupling(func_adj, struct_adj)
            }
            
            logger.info("Created multilayer network representation")
            return multilayer
            
        except Exception as e:
            logger.error(f"Failed to create multilayer graph: {e}")
            return {}
    
    def _compute_interlayer_coupling(self, func_adj: np.ndarray, 
                                   struct_adj: np.ndarray) -> np.ndarray:
        """Compute coupling between functional and structural layers"""
        try:
            # Simple correlation between corresponding edges
            func_edges = func_adj[np.triu_indices_from(func_adj, k=1)]
            struct_edges = struct_adj[np.triu_indices_from(struct_adj, k=1)]
            
            # Node-wise coupling (correlation between connectivity profiles)
            coupling_matrix = np.corrcoef(func_adj, struct_adj)
            n_nodes = func_adj.shape[0]
            
            # Extract cross-modality correlations
            interlayer_coupling = coupling_matrix[:n_nodes, n_nodes:]
            
            return interlayer_coupling
            
        except Exception as e:
            logger.error(f"Failed to compute interlayer coupling: {e}")
            return np.zeros((func_adj.shape[0], func_adj.shape[0]))

class MultimodalGraphIntegrator:
    """Main class for integrating multimodal brain graphs"""
    
    def __init__(self, config: MultimodalConfig = None):
        self.config = config or MultimodalConfig()
        self.validator = MultimodalDataValidator()
        self.feature_integrator = NodeFeatureIntegrator(self.config)
        self.edge_integrator = EdgeIntegrator(self.config)
        self.network_creator = MultimodalNetworkCreator(self.config)
    
    def integrate_graphs(self, functional_graph: Dict, structural_graph: Dict) -> Dict:
        """
        Main method to integrate functional and structural graphs
        
        Args:
            functional_graph: Output from graph_fmri.py
            structural_graph: Output from graph_t1w.py
        
        Returns:
            Dictionary containing integrated multimodal graph
        """
        try:
            logger.info("Starting multimodal graph integration")
            
            # Validate compatibility
            self.validator.validate_graph_compatibility(functional_graph, structural_graph)
            
            # Extract node features
            func_features, func_names = self.feature_integrator.extract_functional_features(functional_graph)
            struct_features, struct_names = self.feature_integrator.extract_structural_features(structural_graph)
            
            # Integrate node features
            integrated_features, integrated_names = self.feature_integrator.integrate_features(
                func_features, struct_features, func_names, struct_names
            )
            
            # Integrate adjacency matrices
            func_adj = functional_graph['adjacency_matrix']
            struct_adj = structural_graph['similarity_matrix']
            integrated_adj = self.edge_integrator.integrate_adjacency_matrices(func_adj, struct_adj)
            
            # Create multimodal graph
            multimodal_graph = self.network_creator.create_multimodal_graph(
                integrated_adj, integrated_features, integrated_names,
                functional_graph['roi_labels']
            )
            
            # Create multilayer representation if requested
            multilayer_graph = self.network_creator.create_multilayer_graph(
                functional_graph, structural_graph
            )
            
            # Prepare output
            integrated_result = {
                'multimodal_adjacency': integrated_adj,
                'multimodal_features': integrated_features,
                'feature_names': integrated_names,
                'multimodal_graph': multimodal_graph,
                'multilayer_graph': multilayer_graph,
                'roi_labels': functional_graph['roi_labels'],
                'n_rois': functional_graph['n_rois'],
                'functional_weight': self.config.functional_weight,
                'structural_weight': self.config.structural_weight,
                'integration_method': self.config.integration_method,
                'edge_integration_method': self.config.edge_integration_method,
                'original_functional': functional_graph,
                'original_structural': structural_graph,
                'config': self.config
            }
            
            logger.info("Multimodal graph integration completed successfully")
            return integrated_result
            
        except Exception as e:
            logger.error(f"Multimodal integration failed: {e}")
            raise

def integrate_multimodal_graphs_cli(functional_graph_file: str,
                                  structural_graph_file: str,
                                  output_file: str = None,
                                  functional_weight: float = 0.6,
                                  structural_weight: float = 0.4,
                                  integration_method: str = 'concatenate',
                                  edge_integration_method: str = 'weighted_sum') -> str:
    """
    Command-line interface for multimodal graph integration
    
    Args:
        functional_graph_file: Path to functional graph pickle file
        structural_graph_file: Path to structural graph pickle file
        output_file: Output file path (optional)
        functional_weight: Weight for functional modality
        structural_weight: Weight for structural modality
        integration_method: Method for integrating features
        edge_integration_method: Method for integrating edges
    
    Returns:
        Path to output file
    """
    try:
        # Load functional graph
        with open(functional_graph_file, 'rb') as f:
            functional_graph = pickle.load(f)
        logger.info(f"Loaded functional graph from: {functional_graph_file}")
        
        # Load structural graph
        with open(structural_graph_file, 'rb') as f:
            structural_graph = pickle.load(f)
        logger.info(f"Loaded structural graph from: {structural_graph_file}")
        
        # Configure integration
        config = MultimodalConfig(
            functional_weight=functional_weight,
            structural_weight=structural_weight,
            integration_method=integration_method,
            edge_integration_method=edge_integration_method
        )
        
        # Integrate graphs
        integrator = MultimodalGraphIntegrator(config)
        multimodal_result = integrator.integrate_graphs(functional_graph, structural_graph)
        
        # Determine output file
        if output_file is None:
            base_name = Path(functional_graph_file).stem.replace('_functional_graph', '')
            output_file = f"{base_name}_multimodal_graph.pkl"
        
        # Save result
        with open(output_file, 'wb') as f:
            pickle.dump(multimodal_result, f)
        
        logger.info(f"Multimodal graph saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"CLI integration failed: {e}")
        raise

def load_and_integrate_from_subject_dir(subject_dir: Path,
                                      functional_file_pattern: str = "*_functional_graph.pkl",
                                      structural_file_pattern: str = "*_structural_graph.pkl") -> Dict:
    """
    Load functional and structural graphs from subject directory and integrate them
    
    Args:
        subject_dir: Directory containing graph files
        functional_file_pattern: Pattern to find functional graph file
        structural_file_pattern: Pattern to find structural graph file
    
    Returns:
        Integrated multimodal graph
    """
    try:
        # Find graph files
        func_files = list(subject_dir.glob(functional_file_pattern))
        struct_files = list(subject_dir.glob(structural_file_pattern))
        
        if not func_files:
            raise FileNotFoundError(f"No functional graph file found in {subject_dir}")
        if not struct_files:
            raise FileNotFoundError(f"No structural graph file found in {subject_dir}")
        
        # Load graphs
        with open(func_files[0], 'rb') as f:
            functional_graph = pickle.load(f)
        
        with open(struct_files[0], 'rb') as f:
            structural_graph = pickle.load(f)
        
        # Integrate
        integrator = MultimodalGraphIntegrator()
        return integrator.integrate_graphs(functional_graph, structural_graph)
        
    except Exception as e:
        logger.error(f"Failed to load and integrate from {subject_dir}: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) >= 3:
        functional_file = sys.argv[1]
        structural_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        result_file = integrate_multimodal_graphs_cli(
            functional_graph_file=functional_file,
            structural_graph_file=structural_file,
            output_file=output_file
        )
        
        print(f"Multimodal graph integrated and saved to: {result_file}")
    else:
        print("Usage: python graph_merge.py <functional_graph.pkl> <structural_graph.pkl> [output_file.pkl]")
        print("Example: python graph_merge.py sub-01_functional_graph.pkl sub-01_structural_graph.pkl sub-01_multimodal.pkl")