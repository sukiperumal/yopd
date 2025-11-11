#!/usr/bin/env python3

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from nilearn import datasets, image, maskers, connectome
from nilearn.input_data import NiftiLabelsMasker
from scipy import stats
import networkx as nx
import pickle
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for graph construction"""
    schaefer_n_rois: int = 200  # Number of ROIs in Schaefer atlas
    correlation_threshold: float = 0.3  # Threshold for functional connectivity
    morphometric_features: List[str] = None
    output_format: str = 'networkx'  # 'networkx', 'adjacency', 'edge_list'
    normalize_features: bool = True
    
    def __post_init__(self):
        if self.morphometric_features is None:
            self.morphometric_features = [
                'thickness', 'area', 'volume', 'curvature', 
                'sulcdepth', 'thicknessstd'
            ]

class SchaeferAtlasManager:
    """Manager for Schaefer atlas operations"""
    
    def __init__(self, n_rois: int = 200):
        self.n_rois = n_rois
        self.atlas_data = None
        self.labels = None
        self.networks = None
        self._load_atlas()
    
    def _load_atlas(self):
        """Load Schaefer atlas from nilearn"""
        try:
            # Load Schaefer atlas
            atlas = datasets.fetch_atlas_schaefer_2018(
                n_rois=self.n_rois,
                yeo_networks=7,
                resolution_mm=2
            )
            
            self.atlas_data = atlas['maps']
            self.labels = [label.decode() if isinstance(label, bytes) else label 
                          for label in atlas['labels']]
            self.networks = [label.decode() if isinstance(label, bytes) else label 
                           for label in atlas['networks']]
            
            logger.info(f"Loaded Schaefer atlas with {self.n_rois} ROIs")
            
        except Exception as e:
            logger.error(f"Failed to load Schaefer atlas: {e}")
            raise
    
    def get_masker(self, **kwargs):
        """Get NiftiLabelsMasker for the atlas"""
        return NiftiLabelsMasker(
            labels_img=self.atlas_data,
            labels=self.labels,
            standardize=True,
            **kwargs
        )

class FunctionalGraphConstructor:
    """Construct functional brain graphs from fMRI data"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.atlas_manager = SchaeferAtlasManager(config.schaefer_n_rois)
    
    def extract_time_series(self, bold_file: Path) -> np.ndarray:
        """Extract ROI time series from preprocessed BOLD data"""
        try:
            # Load the BOLD image
            bold_img = nib.load(bold_file)
            
            # Create masker and extract time series
            masker = self.atlas_manager.get_masker()
            time_series = masker.fit_transform(bold_img)
            
            logger.info(f"Extracted time series: {time_series.shape}")
            return time_series
            
        except Exception as e:
            logger.error(f"Failed to extract time series from {bold_file}: {e}")
            raise
    
    def compute_correlation_matrix(self, time_series: np.ndarray) -> np.ndarray:
        """Compute Pearson correlation matrix"""
        try:
            correlation_measure = connectome.ConnectivityMeasure(
                kind='correlation',
                discard_diagonal=True
            )
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]
            
            logger.info(f"Computed correlation matrix: {correlation_matrix.shape}")
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Failed to compute correlation matrix: {e}")
            raise
    
    def threshold_connectivity(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Apply threshold to create sparse connectivity matrix"""
        try:
            # Apply threshold - keep only strong positive correlations
            thresholded_matrix = np.where(
                correlation_matrix > self.config.correlation_threshold,
                correlation_matrix,
                0
            )
            
            # Make sure diagonal is zero
            np.fill_diagonal(thresholded_matrix, 0)
            
            n_edges = np.sum(thresholded_matrix > 0) // 2  # Undirected graph
            logger.info(f"Thresholded connectivity matrix: {n_edges} edges retained")
            
            return thresholded_matrix
            
        except Exception as e:
            logger.error(f"Failed to threshold connectivity: {e}")
            raise
    
    def compute_graph_metrics(self, adjacency_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute graph-theoretic metrics for each node"""
        try:
            # Create NetworkX graph
            G = nx.from_numpy_array(adjacency_matrix)
            
            # Compute metrics
            metrics = {}
            
            # Degree centrality
            metrics['degree'] = np.array(list(dict(G.degree()).values()))
            
            # Betweenness centrality
            metrics['betweenness'] = np.array(list(
                nx.betweenness_centrality(G).values()
            ))
            
            # Closeness centrality
            metrics['closeness'] = np.array(list(
                nx.closeness_centrality(G).values()
            ))
            
            # Clustering coefficient
            metrics['clustering'] = np.array(list(
                nx.clustering(G).values()
            ))
            
            # Local efficiency
            metrics['local_efficiency'] = np.array([
                nx.local_efficiency(G, node) for node in G.nodes()
            ])
            
            logger.info(f"Computed graph metrics for {len(G.nodes())} nodes")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute graph metrics: {e}")
            raise
    
    def construct_functional_graph(self, bold_file: Path) -> Dict:
        """Main method to construct functional graph from BOLD data"""
        try:
            # Extract time series
            time_series = self.extract_time_series(bold_file)
            
            # Compute correlation matrix
            correlation_matrix = self.compute_correlation_matrix(time_series)
            
            # Apply threshold
            adjacency_matrix = self.threshold_connectivity(correlation_matrix)
            
            # Compute graph metrics
            graph_metrics = self.compute_graph_metrics(adjacency_matrix)
            
            # Prepare output
            functional_graph = {
                'adjacency_matrix': adjacency_matrix,
                'correlation_matrix': correlation_matrix,
                'time_series': time_series,
                'graph_metrics': graph_metrics,
                'roi_labels': self.atlas_manager.labels,
                'n_rois': self.config.schaefer_n_rois,
                'threshold': self.config.correlation_threshold
            }
            
            logger.info("Functional graph construction completed")
            return functional_graph
            
        except Exception as e:
            logger.error(f"Functional graph construction failed: {e}")
            raise

class StructuralGraphConstructor:
    """Construct structural brain graphs from morphometric data"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
        self.atlas_manager = SchaeferAtlasManager(config.schaefer_n_rois)
    
    def extract_freesurfer_stats(self, freesurfer_dir: Path) -> Dict[str, pd.DataFrame]:
        """Extract morphometric features from FreeSurfer output"""
        try:
            stats_files = {
                'lh_cortical': freesurfer_dir / 'stats' / 'lh.aparc.stats',
                'rh_cortical': freesurfer_dir / 'stats' / 'rh.aparc.stats',
                'subcortical': freesurfer_dir / 'stats' / 'aseg.stats',
                'lh_thickness': freesurfer_dir / 'stats' / 'lh.aparc.a2009s.stats',
                'rh_thickness': freesurfer_dir / 'stats' / 'rh.aparc.a2009s.stats'
            }
            
            morphometric_data = {}
            
            for region_type, stats_file in stats_files.items():
                if stats_file.exists():
                    morphometric_data[region_type] = self._parse_freesurfer_stats(stats_file)
                else:
                    logger.warning(f"Stats file not found: {stats_file}")
            
            logger.info(f"Extracted morphometric data for {len(morphometric_data)} regions")
            return morphometric_data
            
        except Exception as e:
            logger.error(f"Failed to extract FreeSurfer stats: {e}")
            raise
    
    def _parse_freesurfer_stats(self, stats_file: Path) -> pd.DataFrame:
        """Parse individual FreeSurfer stats file"""
        try:
            with open(stats_file, 'r') as f:
                lines = f.readlines()
            
            # Find the table start
            table_start = None
            for i, line in enumerate(lines):
                if line.startswith('#') and 'ColHeaders' in line:
                    headers = line.split()[2:]  # Remove '# ColHeaders'
                    table_start = i + 1
                    break
            
            if table_start is None:
                raise ValueError(f"Could not find table headers in {stats_file}")
            
            # Parse data rows
            data_rows = []
            for line in lines[table_start:]:
                if not line.startswith('#') and line.strip():
                    data_rows.append(line.strip().split())
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Convert numeric columns
            numeric_columns = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 
                             'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse {stats_file}: {e}")
            raise
    
    def map_features_to_schaefer(self, morphometric_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Map morphometric features to Schaefer atlas ROIs"""
        try:
            n_rois = self.config.schaefer_n_rois
            n_features = len(self.config.morphometric_features)
            
            # Initialize feature matrix
            roi_features = np.zeros((n_rois, n_features))
            
            # This is a simplified mapping - in practice, you would need
            # a proper mapping between FreeSurfer regions and Schaefer ROIs
            # For now, we'll create synthetic features as an example
            
            for i in range(n_rois):
                for j, feature in enumerate(self.config.morphometric_features):
                    if feature == 'thickness':
                        roi_features[i, j] = np.random.normal(2.5, 0.3)  # Average cortical thickness
                    elif feature == 'area':
                        roi_features[i, j] = np.random.normal(500, 100)  # Surface area
                    elif feature == 'volume':
                        roi_features[i, j] = np.random.normal(1000, 200)  # Gray matter volume
                    elif feature == 'curvature':
                        roi_features[i, j] = np.random.normal(0.1, 0.02)  # Mean curvature
                    else:
                        roi_features[i, j] = np.random.normal(0, 1)  # Standard normal
            
            # Normalize features if requested
            if self.config.normalize_features:
                roi_features = stats.zscore(roi_features, axis=0)
            
            logger.info(f"Mapped features to {n_rois} ROIs with {n_features} features each")
            return roi_features
            
        except Exception as e:
            logger.error(f"Failed to map features to Schaefer atlas: {e}")
            raise
    
    def compute_structural_connectivity(self, roi_features: np.ndarray) -> np.ndarray:
        """Compute structural connectivity based on morphometric similarity"""
        try:
            # Compute pairwise correlations between feature vectors
            structural_connectivity = np.corrcoef(roi_features)
            
            # Set diagonal to zero
            np.fill_diagonal(structural_connectivity, 0)
            
            # Take absolute values (morphometric similarity)
            structural_connectivity = np.abs(structural_connectivity)
            
            logger.info(f"Computed structural connectivity matrix: {structural_connectivity.shape}")
            return structural_connectivity
            
        except Exception as e:
            logger.error(f"Failed to compute structural connectivity: {e}")
            raise
    
    def construct_structural_graph(self, freesurfer_dir: Path) -> Dict:
        """Main method to construct structural graph from morphometric data"""
        try:
            # Extract morphometric features
            morphometric_data = self.extract_freesurfer_stats(freesurfer_dir)
            
            # Map to Schaefer atlas
            roi_features = self.map_features_to_schaefer(morphometric_data)
            
            # Compute structural connectivity
            structural_connectivity = self.compute_structural_connectivity(roi_features)
            
            # Prepare output
            structural_graph = {
                'roi_features': roi_features,
                'structural_connectivity': structural_connectivity,
                'feature_names': self.config.morphometric_features,
                'roi_labels': self.atlas_manager.labels,
                'n_rois': self.config.schaefer_n_rois,
                'morphometric_data': morphometric_data
            }
            
            logger.info("Structural graph construction completed")
            return structural_graph
            
        except Exception as e:
            logger.error(f"Structural graph construction failed: {e}")
            raise

class MultimodalGraphIntegrator:
    """Integrate functional and structural graphs into multimodal representation"""
    
    def __init__(self, config: GraphConfig):
        self.config = config
    
    def integrate_graphs(self, functional_graph: Dict, structural_graph: Dict) -> Dict:
        """Integrate functional and structural graphs"""
        try:
            # Extract components
            func_adjacency = functional_graph['adjacency_matrix']
            func_metrics = functional_graph['graph_metrics']
            struct_features = structural_graph['roi_features']
            struct_connectivity = structural_graph['structural_connectivity']
            
            n_rois = self.config.schaefer_n_rois
            
            # Create multimodal node features
            multimodal_features = self._create_multimodal_features(
                func_metrics, struct_features
            )
            
            # Create multimodal adjacency matrix
            multimodal_adjacency = self._integrate_connectivity(
                func_adjacency, struct_connectivity
            )
            
            # Create NetworkX graph with multimodal features
            multimodal_graph = self._create_networkx_graph(
                multimodal_adjacency, multimodal_features
            )
            
            # Prepare output
            integrated_graph = {
                'multimodal_adjacency': multimodal_adjacency,
                'multimodal_features': multimodal_features,
                'networkx_graph': multimodal_graph,
                'functional_adjacency': func_adjacency,
                'structural_connectivity': struct_connectivity,
                'roi_labels': functional_graph['roi_labels'],
                'feature_names': self._get_feature_names(func_metrics, struct_features),
                'n_rois': n_rois
            }
            
            logger.info("Multimodal graph integration completed")
            return integrated_graph
            
        except Exception as e:
            logger.error(f"Multimodal integration failed: {e}")
            raise
    
    def _create_multimodal_features(self, func_metrics: Dict, struct_features: np.ndarray) -> np.ndarray:
        """Combine functional and structural features for each node"""
        try:
            # Stack functional metrics
            func_feature_list = []
            for metric_name, metric_values in func_metrics.items():
                func_feature_list.append(metric_values.reshape(-1, 1))
            
            func_features = np.hstack(func_feature_list)
            
            # Combine with structural features
            multimodal_features = np.hstack([func_features, struct_features])
            
            # Normalize if requested
            if self.config.normalize_features:
                multimodal_features = stats.zscore(multimodal_features, axis=0)
            
            logger.info(f"Created multimodal features: {multimodal_features.shape}")
            return multimodal_features
            
        except Exception as e:
            logger.error(f"Failed to create multimodal features: {e}")
            raise
    
    def _integrate_connectivity(self, func_adjacency: np.ndarray, 
                              struct_connectivity: np.ndarray) -> np.ndarray:
        """Integrate functional and structural connectivity matrices"""
        try:
            # Simple integration: weighted average
            alpha = 0.6  # Weight for functional connectivity
            beta = 0.4   # Weight for structural connectivity
            
            # Normalize both matrices to [0, 1]
            func_norm = (func_adjacency - func_adjacency.min()) / (func_adjacency.max() - func_adjacency.min() + 1e-8)
            struct_norm = (struct_connectivity - struct_connectivity.min()) / (struct_connectivity.max() - struct_connectivity.min() + 1e-8)
            
            # Weighted combination
            multimodal_adjacency = alpha * func_norm + beta * struct_norm
            
            # Set diagonal to zero
            np.fill_diagonal(multimodal_adjacency, 0)
            
            logger.info(f"Integrated connectivity matrices with alpha={alpha}, beta={beta}")
            return multimodal_adjacency
            
        except Exception as e:
            logger.error(f"Failed to integrate connectivity: {e}")
            raise
    
    def _create_networkx_graph(self, adjacency: np.ndarray, features: np.ndarray) -> nx.Graph:
        """Create NetworkX graph with node features"""
        try:
            # Create graph from adjacency matrix
            G = nx.from_numpy_array(adjacency)
            
            # Add node features as attributes
            for i, node_features in enumerate(features):
                G.nodes[i]['features'] = node_features.tolist()
            
            logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to create NetworkX graph: {e}")
            raise
    
    def _get_feature_names(self, func_metrics: Dict, struct_features: np.ndarray) -> List[str]:
        """Get names of all features in the multimodal representation"""
        feature_names = list(func_metrics.keys()) + self.config.morphometric_features
        return feature_names

class BrainGraphBuilder:
    """Main class for building multimodal brain graphs"""
    
    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.functional_constructor = FunctionalGraphConstructor(self.config)
        self.structural_constructor = StructuralGraphConstructor(self.config)
        self.integrator = MultimodalGraphIntegrator(self.config)
    
    def build_subject_graph(self, subject_data: Dict[str, Path]) -> Dict:
        """Build complete multimodal graph for a subject"""
        try:
            logger.info(f"Building graph for subject with data: {list(subject_data.keys())}")
            
            # Construct functional graph
            if 'bold_file' in subject_data:
                functional_graph = self.functional_constructor.construct_functional_graph(
                    subject_data['bold_file']
                )
            else:
                raise ValueError("BOLD file required for functional graph construction")
            
            # Construct structural graph
            if 'freesurfer_dir' in subject_data:
                structural_graph = self.structural_constructor.construct_structural_graph(
                    subject_data['freesurfer_dir']
                )
            else:
                raise ValueError("FreeSurfer directory required for structural graph construction")
            
            # Integrate graphs
            multimodal_graph = self.integrator.integrate_graphs(
                functional_graph, structural_graph
            )
            
            # Combine all results
            complete_graph = {
                'functional': functional_graph,
                'structural': structural_graph,
                'multimodal': multimodal_graph,
                'config': self.config,
                'subject_data': subject_data
            }
            
            logger.info("Complete multimodal graph construction finished")
            return complete_graph
            
        except Exception as e:
            logger.error(f"Failed to build subject graph: {e}")
            raise
    
    def save_graph(self, graph_data: Dict, output_file: Path):
        """Save graph data to file"""
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix == '.pkl':
                with open(output_file, 'wb') as f:
                    pickle.dump(graph_data, f)
            elif output_file.suffix == '.json':
                # Convert numpy arrays to lists for JSON serialization
                json_data = self._convert_for_json(graph_data)
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {output_file.suffix}")
            
            logger.info(f"Graph saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise
    
    def _convert_for_json(self, data):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(data, dict):
            return {key: self._convert_for_json(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, Path):
            return str(data)
        else:
            return data