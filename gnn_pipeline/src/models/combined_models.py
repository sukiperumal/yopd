#!/usr/bin/env python3
"""
Combined GCN+GAT Models for Brain Graph Classification
======================================================

Sequential and parallel combinations of GCN and GAT architectures for
enhanced brain connectivity analysis.

Models:
1. Sequential: GCN → GAT pipeline with feature progression
2. Parallel: GCN + GAT branches with feature fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BrainGCNGAT_Sequential(nn.Module):
    """
    Sequential GCN → GAT architecture
    
    Pipeline:
    1. GCN layers for spatial feature smoothing
    2. GAT layers for attention-based refinement
    3. Global pooling → Classification
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: list = [64, 32],
        dropout: float = 0.4,
        gat_heads: int = 4,
        attention_dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        super(BrainGCNGAT_Sequential, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling
        
        if len(hidden_channels) != 2:
            raise ValueError("Sequential model requires exactly 2 hidden channel sizes")
        
        hidden_dim1, hidden_dim2 = hidden_channels
        
        # GCN layers (spatial smoothing)
        self.gcn1 = GCNConv(num_features, hidden_dim1)
        self.gcn2 = GCNConv(hidden_dim1, hidden_dim1)  # Keep dimension for GAT
        
        # GAT layers (attention refinement)
        self.gat1 = GATConv(
            hidden_dim1, 
            hidden_dim2, 
            heads=gat_heads,
            dropout=attention_dropout,
            concat=True
        )
        self.gat2 = GATConv(
            hidden_dim2 * gat_heads,
            hidden_dim2,
            heads=1,
            dropout=attention_dropout,
            concat=False
        )
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2 // 2, num_classes)
        )
        
        # Store attention weights
        self.attention_weights = None
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data: Batch, return_attention: bool = False) -> torch.Tensor:
        """Forward pass through sequential GCN → GAT"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        
        # GCN phase: Spatial feature smoothing
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # GAT phase: Attention-based refinement
        if return_attention:
            x, (edge_idx1, att1) = self.gat1(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x = self.gat1(x, edge_index, edge_attr=edge_attr)
            
        x = F.elu(x)
        x = self.dropout3(x)
        
        if return_attention:
            x, (edge_idx2, att2) = self.gat2(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            self.attention_weights = {
                'gat_layer1': (edge_idx1, att1),
                'gat_layer2': (edge_idx2, att2)
            }
        else:
            x = self.gat2(x, edge_index, edge_attr=edge_attr)
            
        x = F.elu(x)
        x = self.dropout4(x)
        
        # Global pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention:
            return logits, self.attention_weights
        else:
            return logits


class BrainGCNGAT_Parallel(nn.Module):
    """
    Parallel GCN + GAT architecture with feature fusion
    
    Architecture:
    1. Parallel GCN and GAT branches
    2. Feature fusion (concatenation or attention)
    3. Final classification layers
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_channels: list = [64, 32],
        dropout: float = 0.4,
        gat_heads: int = 4,
        fusion_method: str = 'concat',  # 'concat', 'add', 'attention'
        pooling: str = 'mean'
    ):
        super(BrainGCNGAT_Parallel, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.fusion_method = fusion_method
        self.pooling = pooling
        
        if len(hidden_channels) != 2:
            raise ValueError("Parallel model requires exactly 2 hidden channel sizes")
        
        hidden_dim1, hidden_dim2 = hidden_channels
        
        # GCN Branch
        self.gcn1 = GCNConv(num_features, hidden_dim1)
        self.gcn2 = GCNConv(hidden_dim1, hidden_dim2)
        
        # GAT Branch  
        self.gat1 = GATConv(
            num_features,
            hidden_dim1 // gat_heads,  # Adjust for head concatenation
            heads=gat_heads,
            dropout=0.1,
            concat=True
        )
        self.gat2 = GATConv(
            hidden_dim1,
            hidden_dim2,
            heads=1,
            dropout=0.1,
            concat=False
        )
        
        # Dropout layers
        self.gcn_dropout1 = nn.Dropout(dropout)
        self.gcn_dropout2 = nn.Dropout(dropout)
        self.gat_dropout1 = nn.Dropout(dropout)
        self.gat_dropout2 = nn.Dropout(dropout)
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = hidden_dim2 * 2
        elif fusion_method == 'add':
            fusion_dim = hidden_dim2
        elif fusion_method == 'attention':
            fusion_dim = hidden_dim2
            self.attention_fusion = nn.MultiheadAttention(
                embed_dim=hidden_dim2,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        # Store attention weights
        self.attention_weights = None
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, data: Batch, return_attention: bool = False) -> torch.Tensor:
        """Forward pass through parallel GCN + GAT"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        
        # GCN Branch
        x_gcn = self.gcn1(x, edge_index)
        x_gcn = F.relu(x_gcn)
        x_gcn = self.gcn_dropout1(x_gcn)
        
        x_gcn = self.gcn2(x_gcn, edge_index)
        x_gcn = F.relu(x_gcn)
        x_gcn = self.gcn_dropout2(x_gcn)
        
        # GAT Branch
        if return_attention:
            x_gat, (edge_idx1, att1) = self.gat1(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            x_gat = self.gat1(x, edge_index, edge_attr=edge_attr)
            
        x_gat = F.elu(x_gat)
        x_gat = self.gat_dropout1(x_gat)
        
        if return_attention:
            x_gat, (edge_idx2, att2) = self.gat2(
                x_gat, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            self.attention_weights = {
                'gat_layer1': (edge_idx1, att1),
                'gat_layer2': (edge_idx2, att2)
            }
        else:
            x_gat = self.gat2(x_gat, edge_index, edge_attr=edge_attr)
            
        x_gat = F.elu(x_gat)
        x_gat = self.gat_dropout2(x_gat)
        
        # Global pooling for both branches
        if self.pooling == 'mean':
            x_gcn = global_mean_pool(x_gcn, batch)
            x_gat = global_mean_pool(x_gat, batch)
        elif self.pooling == 'max':
            x_gcn = global_max_pool(x_gcn, batch)
            x_gat = global_max_pool(x_gat, batch)
        
        # Feature fusion
        if self.fusion_method == 'concat':
            x_fused = torch.cat([x_gcn, x_gat], dim=1)
        elif self.fusion_method == 'add':
            x_fused = x_gcn + x_gat
        elif self.fusion_method == 'attention':
            # Stack features for attention
            stacked = torch.stack([x_gcn, x_gat], dim=1)  # [batch, 2, hidden]
            x_fused, _ = self.attention_fusion(stacked, stacked, stacked)
            x_fused = x_fused.mean(dim=1)  # Average attention output
        
        # Classification
        logits = self.classifier(x_fused)
        
        if return_attention:
            return logits, self.attention_weights
        else:
            return logits


def demonstrate_combined_models():
    """Demonstrate combined GCN+GAT models"""
    print("🧠 Combined GCN+GAT Models Demo")
    print("="*35)
    
    # Sequential model
    seq_model = BrainGCNGAT_Sequential(
        num_features=17,
        num_classes=3,
        hidden_channels=[64, 32],
        dropout=0.4,
        gat_heads=4
    )
    
    # Parallel model
    par_model = BrainGCNGAT_Parallel(
        num_features=17,
        num_classes=3,
        hidden_channels=[64, 32],
        dropout=0.4,
        gat_heads=4,
        fusion_method='concat'
    )
    
    print(f"✅ Combined models created")
    
    # Model statistics
    seq_params = sum(p.numel() for p in seq_model.parameters())
    par_params = sum(p.numel() for p in par_model.parameters())
    
    print(f"\n📊 Sequential GCN→GAT:")
    print(f"   Architecture: GCN(17→64) → GCN(64→64) → GAT(64→32) → GAT(32→32)")
    print(f"   Parameters: {seq_params:,}")
    print(f"   Features: Spatial smoothing → Attention refinement")
    
    print(f"\n📊 Parallel GCN+GAT:")
    print(f"   Architecture: GCN branch + GAT branch → Fusion → Classifier")
    print(f"   Parameters: {par_params:,}")
    print(f"   Features: Complementary representations → Feature fusion")
    
    print(f"\n🎯 Key advantages:")
    print(f"   ✓ Sequential: Progressive feature refinement")
    print(f"   ✓ Parallel: Complementary feature extraction")
    print(f"   ✓ Both: Attention weight extraction for explainability")
    print(f"   ✓ Both: Configurable fusion and pooling strategies")


if __name__ == "__main__":
    demonstrate_combined_models()