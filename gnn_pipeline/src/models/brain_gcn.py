#!/usr/bin/env python3
"""
Graph Convolutional Network (GCN) for Brain Graph Classification
================================================================

GCN implementation with spatial feature smoothing for multimodal brain 
connectivity analysis.

Architecture:
- Layer 1: GCNConv(F, 64) → ReLU → Dropout
- Layer 2: GCNConv(64, 32) → ReLU → Dropout  
- Global pooling → Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BrainGCN(nn.Module):
    """
    Graph Convolutional Network for brain graph classification
    
    Features:
    - Spatial feature smoothing across brain connectivity
    - Configurable hidden dimensions and dropout
    - Multiple global pooling strategies
    - Xavier weight initialization
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: list = [64, 32],
        dropout: float = 0.5,
        pooling: str = 'mean'
    ):
        super(BrainGCN, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling
        
        # Validate hidden channels
        if len(hidden_channels) != 2:
            raise ValueError("BrainGCN requires exactly 2 hidden channel sizes")
        
        hidden_dim1, hidden_dim2 = hidden_channels
        
        # GCN layers
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2 // 2, num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights with Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass through GCN
        
        Args:
            data: PyG Batch object containing:
                - x: Node features [N, F]
                - edge_index: Edge connectivity [2, E]
                - batch: Batch assignment [N]
                
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_node_embeddings(self, data: Batch) -> torch.Tensor:
        """Get node embeddings after GCN layers (before pooling)"""
        x, edge_index = data.x, data.edge_index
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GCN layer  
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        return x


def brain_gcn():
    # Create model
    model = BrainGCN(
        num_features=17,  # Multimodal brain features
        num_classes=3,    # HC, PIGD, TDPD
        hidden_channels=[64, 32],
        dropout=0.5,
        pooling='mean'
    )
    
    print(f"✅ BrainGCN model created")
    print(f"📊 Architecture:")
    print(f"   Input features: 17 (multimodal)")
    print(f"   Hidden layers: 64 → 32")
    print(f"   Output classes: 3 (HC, PIGD, TDPD)")
    print(f"   Dropout rate: 0.5")
    print(f"   Pooling: mean")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024:.1f} KB")
    
    print(f"\n🎯 Key features:")
    print(f"   ✓ Spatial feature smoothing via graph convolutions")
    print(f"   ✓ Xavier weight initialization for stable training")
    print(f"   ✓ Configurable architecture and dropout")
    print(f"   ✓ Multiple global pooling strategies")
    print(f"   ✓ Node embedding extraction for analysis")


if __name__ == "__main__":
    brain_gcn()