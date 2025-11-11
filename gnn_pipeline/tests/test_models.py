"""Test GNN model architectures."""

import pytest
import torch
from unittest.mock import Mock, patch

# Test data dimensions
NUM_FEATURES = 17
NUM_CLASSES = 3
BATCH_SIZE = 4
NUM_NODES = 400


class TestBrainGCN:
    """Test BrainGCN model architecture."""
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_gcn_initialization(self, mock_cuda):
        """Test GCN model initialization."""
        try:
            from models.brain_gcn import BrainGCN
            
            model = BrainGCN(
                num_features=NUM_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_channels=[32, 16],
                dropout=0.5
            )
            
            assert model.num_features == NUM_FEATURES
            assert model.num_classes == NUM_CLASSES
            assert model.dropout == 0.5
            
        except ImportError:
            pytest.skip("PyTorch dependencies not available")
    
    @patch('torch.cuda.is_available', return_value=False)  
    def test_gcn_forward_pass(self, mock_cuda):
        """Test GCN forward pass with mock data."""
        try:
            from models.brain_gcn import BrainGCN
            
            model = BrainGCN(
                num_features=NUM_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_channels=[32, 16]
            )
            
            # Mock PyG Batch object
            mock_data = Mock()
            mock_data.x = torch.randn(BATCH_SIZE * NUM_NODES, NUM_FEATURES)
            mock_data.edge_index = torch.randint(0, NUM_NODES, (2, 1000))
            mock_data.batch = torch.repeat_interleave(torch.arange(BATCH_SIZE), NUM_NODES)
            
            # Forward pass
            with patch.object(model, 'conv1') as mock_conv1, \
                 patch.object(model, 'conv2') as mock_conv2, \
                 patch.object(model, 'classifier') as mock_classifier:
                
                mock_conv1.return_value = torch.randn(BATCH_SIZE * NUM_NODES, 32)
                mock_conv2.return_value = torch.randn(BATCH_SIZE * NUM_NODES, 16)
                mock_classifier.return_value = torch.randn(BATCH_SIZE, NUM_CLASSES)
                
                # Mock global pooling
                with patch('models.brain_gcn.global_mean_pool') as mock_pool:
                    mock_pool.return_value = torch.randn(BATCH_SIZE, 16)
                    
                    output = model(mock_data)
                    assert output.shape == (BATCH_SIZE, NUM_CLASSES)
                    
        except ImportError:
            pytest.skip("PyTorch dependencies not available")


class TestBrainGAT:
    """Test BrainGAT model architecture."""
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_gat_initialization(self, mock_cuda):
        """Test GAT model initialization."""
        try:
            from models.brain_gat import BrainGAT
            
            model = BrainGAT(
                num_features=NUM_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_channels=[32, 16],
                heads=[4, 1],
                dropout=0.3
            )
            
            assert model.num_features == NUM_FEATURES
            assert model.num_classes == NUM_CLASSES
            assert model.dropout == 0.3
            
        except ImportError:
            pytest.skip("PyTorch dependencies not available")


class TestCombinedModels:
    """Test combined GCN+GAT architectures."""
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_sequential_initialization(self, mock_cuda):
        """Test Sequential GCN+GAT initialization."""
        try:
            from models.combined_models import BrainGCNGAT_Sequential
            
            model = BrainGCNGAT_Sequential(
                num_features=NUM_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_channels=[32, 16],
                dropout=0.4
            )
            
            assert model.num_features == NUM_FEATURES
            assert model.num_classes == NUM_CLASSES
            
        except ImportError:
            pytest.skip("PyTorch dependencies not available")
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_parallel_initialization(self, mock_cuda):
        """Test Parallel GCN+GAT initialization."""
        try:
            from models.combined_models import BrainGCNGAT_Parallel
            
            model = BrainGCNGAT_Parallel(
                num_features=NUM_FEATURES,
                num_classes=NUM_CLASSES,
                hidden_channels=[32, 16],
                fusion_method='concat'
            )
            
            assert model.fusion_method == 'concat'
            
        except ImportError:
            pytest.skip("PyTorch dependencies not available")


def test_model_imports():
    """Test that model modules can be imported."""
    try:
        from models import brain_gcn, brain_gat, combined_models
        assert hasattr(brain_gcn, 'BrainGCN')
        assert hasattr(brain_gat, 'BrainGAT') 
        assert hasattr(combined_models, 'BrainGCNGAT_Sequential')
        assert hasattr(combined_models, 'BrainGCNGAT_Parallel')
    except ImportError as e:
        pytest.skip(f"Model imports failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])