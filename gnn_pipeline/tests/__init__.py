"""Test suite for Brain GNN pipeline."""

import pytest
import sys
from pathlib import Path

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test configuration
TEST_CONFIG = {
    "data": {
        "num_synthetic_subjects": 10,
        "test_split": 0.2,
        "val_split": 0.15
    },
    "training": {
        "num_epochs": 5,
        "cv_folds": 2,
        "batch_size": 4
    },
    "models": {
        "gcn": {
            "hidden_channels": [16, 8],
            "dropout": 0.2
        }
    }
}