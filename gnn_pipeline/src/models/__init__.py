"""GNN model architectures for brain connectivity analysis."""

from .brain_gcn import BrainGCN
from .brain_gat import BrainGAT
from .combined_models import BrainGCNGAT_Sequential, BrainGCNGAT_Parallel

__all__ = [
    "BrainGCN", 
    "BrainGAT", 
    "BrainGCNGAT_Sequential", 
    "BrainGCNGAT_Parallel"
]