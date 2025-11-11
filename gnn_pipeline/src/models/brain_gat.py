#!/usr/bin/env python3
"""
Graph Attention Network (GAT) for Brain Graph Classification
============================================================

GAT implementation with multi-head attention and weight extraction for 
explainable brain connectivity analysis.

Architecture:
- Layer 1: GAT(F, 64, heads=8) → ELU → Dropout
- Layer 2: GAT(64*8, 32, heads=1) → ELU → Dropout
- Global pooling → Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BrainGAT(nn.Module):
    """
    Graph Attention Network for brain graph classification
    
    Features:
    - Multi-head attention for learning edge importance
    - Attention weight extraction for explainability
    - Configurable heads and hidden dimensions
    - Edge feature incorporation
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: list = [64, 32],
        heads: list = [8, 1],
        dropout: float = 0.3,
        edge_dim: Optional[int] = None,
        attention_dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        super(BrainGAT, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.pooling = pooling
        
        # Validate inputs
        if len(hidden_channels) != 2 or len(heads) != 2:
            raise ValueError("BrainGAT requires exactly 2 hidden channels and 2 head specifications")
        
        hidden_dim1, hidden_dim2 = hidden_channels
        heads1, heads2 = heads
        
        # GAT layers
        self.conv1 = GATConv(
            num_features, 
            hidden_dim1, 
            heads=heads1,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            concat=True
        )
        
        self.conv2 = GATConv(
            hidden_dim1 * heads1,  # Concatenated output from previous layer
            hidden_dim2,
            heads=heads2,
            dropout=attention_dropout,
            edge_dim=edge_dim,
            concat=False  # Average for final layer
        )
        
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
        
        # Store for attention extraction
        self.attention_weights = None
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data: Batch, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through GAT
        
        Args:
            data: PyG Batch object containing:
                - x: Node features [N, F]
                - edge_index: Edge connectivity [2, E]
                - edge_attr: Edge features [E, D] (optional)
                - batch: Batch assignment [N]
            return_attention: Whether to return attention weights
                
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        
        # First GAT layer with attention extraction
        if return_attention:
            x, (edge_index_att, att1) = self.conv1(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
            
        x = F.elu(x)
        x = self.dropout1(x)
        
        # Second GAT layer with attention extraction
        if return_attention:
            x, (edge_index_att2, att2) = self.conv2(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            # Store attention weights for explainability
            self.attention_weights = {
                'layer1': (edge_index_att, att1),
                'layer2': (edge_index_att2, att2)
            }
        else:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
            
        x = F.elu(x)
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
        
        if return_attention:
            return logits, self.attention_weights
        else:
            return logits
    
    def get_attention_weights(self, data: Batch) -> Tuple[torch.Tensor, dict]:
        """
        Get predictions and attention weights for explainability
        
        Returns:
            logits: Model predictions
            attention_weights: Dictionary with attention weights for each layer
        """
        return self.forward(data, return_attention=True)
    
    def get_node_embeddings(self, data: Batch) -> torch.Tensor:
        """Get node embeddings after GAT layers (before pooling)"""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # First GAT layer
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        return x


def demonstrate_brain_gat():
    """Demonstrate BrainGAT model"""
    print("🧠 BrainGAT Model Demo")
    print("="*25)
    
    # Create model
    model = BrainGAT(
        num_features=17,     # Multimodal brain features
        num_classes=3,       # HC, PIGD, TDPD
        hidden_channels=[64, 32],
        heads=[8, 1],        # 8 heads in first layer, 1 in second
        dropout=0.3,
        attention_dropout=0.1,
        pooling='mean'
    )
    
    print(f"✅ BrainGAT model created")
    print(f"📊 Architecture:")
    print(f"   Input features: 17 (multimodal)")
    print(f"   Hidden layers: 64 → 32")
    print(f"   Attention heads: 8 → 1")
    print(f"   Output classes: 3 (HC, PIGD, TDPD)")
    print(f"   Dropout rate: 0.3")
    print(f"   Attention dropout: 0.1")
    print(f"   Pooling: mean")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024:.1f} KB")
    
    print(f"\n🎯 Key features:")
    print(f"   ✓ Multi-head attention for edge importance learning")
    print(f"   ✓ Attention weight extraction for explainability")
    print(f"   ✓ Edge feature incorporation capability")
    print(f"   ✓ Configurable attention heads and dropout")
    print(f"   ✓ Node embedding extraction for analysis")


if __name__ == "__main__":
    demonstrate_brain_gat()