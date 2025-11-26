#!/usr/bin/env python3
"""
Data Loading and Preprocessing for Brain GNN Classification
===========================================================

Provides data loading utilities for brain graph data, including:
- Loading comprehensive graphs from JSON files
- Creating PyTorch Geometric Data objects
- Train/val/test splitting with stratification
- Data augmentation options

Author: Generated for YOPD Brain Graph Analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for data loading and training."""
    
    comprehensive_graphs_dir: str = "outputs"
    num_synthetic_subjects: int = 100
    test_split: float = 0.2
    val_split: float = 0.15
    random_seed: int = 42
    stratify: bool = True
    balance_classes: bool = True
    batch_size: int = 16


class BrainGraphDataset:
    """Dataset class for brain graph data."""
    
    # Class label mapping
    LABEL_MAP = {
        'HC': 0,
        'PIGD': 1,
        'TDPD': 2,
        'hc': 0,
        'pigd': 1, 
        'tdpd': 2,
        'Healthy Control': 0,
        'PIGD-PD': 1,
        'TD-PD': 2
    }
    
    def __init__(self, config: TrainingConfig):
        """Initialize dataset with configuration.
        
        Args:
            config: TrainingConfig with data loading parameters
        """
        self.config = config
        self.graphs_dir = Path(config.comprehensive_graphs_dir)
        self.data_list: List[Data] = []
        self.labels: List[int] = []
        self.class_counts: Dict[int, int] = {}
        
        logger.info(f"Initialized BrainGraphDataset")
        logger.info(f"  Graphs directory: {self.graphs_dir}")
    
    def load_and_process(self) -> List[Data]:
        """Load and process all brain graphs.
        
        Returns:
            List of PyTorch Geometric Data objects
        """
        logger.info("Loading and processing brain graphs...")
        
        # Find all graph files
        graph_files = self._find_graph_files()
        
        if not graph_files:
            logger.warning(f"No graph files found in {self.graphs_dir}")
            logger.info("Creating synthetic data for demonstration")
            return self._create_synthetic_data()
        
        logger.info(f"Found {len(graph_files)} graph files")
        
        # Load each graph
        for graph_file in graph_files:
            try:
                data = self._load_single_graph(graph_file)
                if data is not None:
                    self.data_list.append(data)
                    self.labels.append(data.y.item())
            except Exception as e:
                logger.warning(f"Error loading {graph_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.data_list)} graphs successfully")
        
        # Compute class distribution
        self._compute_class_distribution()
        
        return self.data_list
    
    def _find_graph_files(self) -> List[Path]:
        """Find all graph JSON files in the graphs directory."""
        graph_files = []
        
        if not self.graphs_dir.exists():
            logger.warning(f"Graphs directory does not exist: {self.graphs_dir}")
            return graph_files
        
        # Look for comprehensive graph files
        patterns = [
            "**/comprehensive_graph*.json",
            "**/graph_*.json",
            "**/brain_graph*.json",
            "**/*_graph.json"
        ]
        
        for pattern in patterns:
            files = list(self.graphs_dir.glob(pattern))
            graph_files.extend(files)
        
        # Remove duplicates
        graph_files = list(set(graph_files))
        
        return sorted(graph_files)
    
    def _load_single_graph(self, graph_file: Path) -> Optional[Data]:
        """Load a single graph from JSON file.
        
        Args:
            graph_file: Path to the graph JSON file
            
        Returns:
            PyTorch Geometric Data object or None if loading fails
        """
        with open(graph_file, 'r') as f:
            graph_data = json.load(f)
        
        # Extract node features
        nodes = graph_data.get('nodes', [])
        if not nodes:
            logger.warning(f"No nodes found in {graph_file}")
            return None
        
        # Build node feature matrix
        node_features = []
        for node in nodes:
            features = self._extract_node_features(node)
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Extract edges
        edges = graph_data.get('edges', [])
        if not edges:
            # Create fully connected graph if no edges
            num_nodes = len(nodes)
            edge_index = self._create_fully_connected_edges(num_nodes)
            edge_attr = None
        else:
            edge_index, edge_attr = self._extract_edges(edges, len(nodes))
        
        # Get label
        label = self._extract_label(graph_data, graph_file)
        y = torch.tensor([label], dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        if edge_attr is not None:
            data.edge_attr = edge_attr
        
        # Store metadata
        data.graph_id = graph_file.stem
        data.num_nodes = x.size(0)
        data.num_edges = edge_index.size(1)
        
        return data
    
    def _extract_node_features(self, node: Dict[str, Any]) -> List[float]:
        """Extract features from a node dictionary.
        
        Args:
            node: Node dictionary with features
            
        Returns:
            List of numerical features
        """
        features = []
        
        # Common feature keys to extract
        feature_keys = [
            'degree', 'clustering_coefficient', 'betweenness_centrality',
            'closeness_centrality', 'pagerank', 'eigenvector_centrality',
            'local_efficiency', 'volume', 'thickness', 'surface_area',
            'mean_curvature', 'gaussian_curvature', 'folding_index',
            'curvature_index', 'gray_matter_volume', 'white_matter_volume',
            'csf_volume'
        ]
        
        for key in feature_keys:
            if key in node:
                val = node[key]
                if isinstance(val, (int, float)):
                    features.append(float(val))
                elif isinstance(val, list) and len(val) > 0:
                    features.append(float(val[0]))
                else:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return features
    
    def _create_fully_connected_edges(self, num_nodes: int) -> torch.Tensor:
        """Create fully connected edge index.
        
        Note: This creates O(n²) edges which can be memory-intensive for large graphs.
        """
        if num_nodes > 500:
            logger.warning(f"Creating fully connected graph with {num_nodes} nodes "
                         f"will produce ~{num_nodes * (num_nodes - 1)} edges. "
                         "Consider using sparse connectivity for better memory efficiency.")
        row = []
        col = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    row.append(i)
                    col.append(j)
        return torch.tensor([row, col], dtype=torch.long)
    
    def _extract_edges(self, edges: List[Dict], num_nodes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract edge index and attributes from edge list.
        
        Args:
            edges: List of edge dictionaries
            num_nodes: Number of nodes in the graph
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        row = []
        col = []
        weights = []
        
        for edge in edges:
            source = edge.get('source', edge.get('from', edge.get('src')))
            target = edge.get('target', edge.get('to', edge.get('dst')))
            weight = edge.get('weight', edge.get('value', 1.0))
            
            if source is not None and target is not None:
                # Handle both string and integer indices
                if isinstance(source, str):
                    # Use deterministic hash for string-to-index conversion
                    source = int(source) if source.isdigit() else (sum(ord(c) for c in source) % num_nodes)
                if isinstance(target, str):
                    # Use deterministic hash for string-to-index conversion
                    target = int(target) if target.isdigit() else (sum(ord(c) for c in target) % num_nodes)
                
                if 0 <= source < num_nodes and 0 <= target < num_nodes:
                    row.append(source)
                    col.append(target)
                    weights.append(float(weight) if isinstance(weight, (int, float)) else 1.0)
        
        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1) if weights else None
        
        return edge_index, edge_attr
    
    def _extract_label(self, graph_data: Dict, graph_file: Path) -> int:
        """Extract label from graph data or filename.
        
        Args:
            graph_data: Graph dictionary
            graph_file: Path to graph file
            
        Returns:
            Integer label
        """
        # Try to get label from graph data
        label_str = graph_data.get('label', graph_data.get('class', graph_data.get('group')))
        
        if label_str is not None:
            if isinstance(label_str, int):
                return label_str
            if label_str in self.LABEL_MAP:
                return self.LABEL_MAP[label_str]
        
        # Try to extract from filename
        filename = graph_file.stem.lower()
        for label_key, label_val in self.LABEL_MAP.items():
            if label_key.lower() in filename:
                return label_val
        
        # Default to HC (0) if no label found
        logger.warning(f"No label found for {graph_file}, defaulting to HC")
        return 0
    
    def _compute_class_distribution(self):
        """Compute class distribution of loaded data."""
        self.class_counts = {}
        for label in self.labels:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
        
        logger.info(f"Class distribution: {self.class_counts}")
    
    def _create_synthetic_data(self) -> List[Data]:
        """Create synthetic brain graph data for demonstration.
        
        Returns:
            List of synthetic PyTorch Geometric Data objects
        """
        np.random.seed(self.config.random_seed)
        
        num_subjects = self.config.num_synthetic_subjects
        num_nodes = 400  # Typical number of ROIs
        num_features = 17
        num_classes = 3
        
        logger.info(f"Creating {num_subjects} synthetic graphs")
        
        for i in range(num_subjects):
            # Random node features
            x = torch.randn(num_nodes, num_features)
            
            # Random edges (sparse connectivity, no self-loops)
            num_edges = num_nodes * 10
            row = torch.randint(0, num_nodes, (num_edges * 2,))
            col = torch.randint(0, num_nodes, (num_edges * 2,))
            
            # Remove self-loops
            mask = row != col
            row = row[mask][:num_edges]
            col = col[mask][:num_edges]
            edge_index = torch.stack([row, col], dim=0)
            
            # Random label
            y = torch.tensor([i % num_classes], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data.graph_id = f"synthetic_{i:04d}"
            
            self.data_list.append(data)
            self.labels.append(y.item())
        
        self._compute_class_distribution()
        
        return self.data_list
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders with stratified splitting.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        from sklearn.model_selection import train_test_split
        
        if not self.data_list:
            self.load_and_process()
        
        indices = list(range(len(self.data_list)))
        
        # First split: train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.config.test_split,
            random_state=self.config.random_seed,
            stratify=[self.labels[i] for i in indices] if self.config.stratify else None
        )
        
        # Second split: train vs val
        val_size = self.config.val_split / (1 - self.config.test_split)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size,
            random_state=self.config.random_seed,
            stratify=[self.labels[i] for i in train_val_idx] if self.config.stratify else None
        )
        
        # Create data loaders
        train_data = [self.data_list[i] for i in train_idx]
        val_data = [self.data_list[i] for i in val_idx]
        test_data = [self.data_list[i] for i in test_idx]
        
        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False)
        
        logger.info(f"Created data loaders: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return train_loader, val_loader, test_loader
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the class distribution of loaded data.
        
        Returns:
            Dictionary mapping class labels to counts
        """
        return self.class_counts.copy()
    
    def get_num_features(self) -> int:
        """Get number of node features.
        
        Returns:
            Number of features per node
        """
        if self.data_list:
            return self.data_list[0].x.size(1)
        return 17  # Default
    
    def get_num_classes(self) -> int:
        """Get number of classes.
        
        Returns:
            Number of unique classes
        """
        return len(set(self.labels)) if self.labels else 3


def demonstrate_data_loading():
    """Demonstrate data loading functionality."""
    print("🧠 Brain Graph Data Loader Demo")
    print("=" * 40)
    
    config = TrainingConfig(
        comprehensive_graphs_dir="outputs",
        num_synthetic_subjects=50
    )
    
    dataset = BrainGraphDataset(config)
    data_list = dataset.load_and_process()
    
    print(f"✅ Loaded {len(data_list)} graphs")
    print(f"📊 Class distribution: {dataset.get_class_distribution()}")
    
    if data_list:
        sample = data_list[0]
        print(f"📈 Sample graph stats:")
        print(f"   - Nodes: {sample.num_nodes}")
        print(f"   - Features per node: {sample.x.size(1)}")
        print(f"   - Edges: {sample.edge_index.size(1)}")
        print(f"   - Label: {sample.y.item()}")


if __name__ == "__main__":
    demonstrate_data_loading()
