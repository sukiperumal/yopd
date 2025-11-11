"""
Brain GNN Classification Pipeline
================================

A comprehensive Graph Neural Network pipeline for classifying Young-Onset 
Parkinson's Disease (YOPD) subtypes using multimodal brain connectivity data.

Modules:
    data: Data loading and preprocessing
    models: GNN model architectures  
    training: Training framework and evaluation
    explainability: Model interpretation and analysis
    visualization: Comprehensive plotting and reporting
"""

__version__ = "1.0.0"
__author__ = "YOPD Research Team"

from . import data, models, training, explainability, visualization

__all__ = ["data", "models", "training", "explainability", "visualization"]