#!/usr/bin/env python3
"""
Main entry point for Brain GNN Classification Pipeline
======================================================

This script provides command-line interface for the complete pipeline execution.

Usage:
    python main.py --config configs/default_config.json
    python main.py --demo
    python main.py --model gcn --data-dir /path/to/data
"""

import sys
import argparse
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.run_pipeline import BrainGNNPipeline, create_default_config, run_demonstration


def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Brain GNN Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --demo                                    # Run demonstration
    python main.py --config configs/default_config.json     # Full pipeline
    python main.py --config configs/dev_config.json         # Development run
    python main.py --model gcn --epochs 100                 # Quick GCN training
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true', 
        help='Run demonstration mode with mock data'
    )
    
    parser.add_argument(
        '--model',
        choices=['gcn', 'gat', 'sequential', 'parallel', 'all'],
        help='Specific model to train (overrides config)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing brain graph data (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        help='Number of cross-validation folds (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Handle demonstration mode
    if args.demo:
        print("🧠 Running Brain GNN Pipeline Demonstration")
        print("=" * 50)
        run_demonstration()
        return
    
    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from {config_path}")
    else:
        # Use default configuration
        config = create_default_config()
        print("✅ Using default configuration")
    
    # Apply command-line overrides
    if args.data_dir:
        config['data']['comprehensive_graphs_dir'] = args.data_dir
        print(f"📁 Data directory: {args.data_dir}")
    
    if args.output_dir:
        config['output']['base_dir'] = args.output_dir
        print(f"📁 Output directory: {args.output_dir}")
    
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        print(f"🔄 Training epochs: {args.epochs}")
    
    if args.cv_folds:
        config['training']['cv_folds'] = args.cv_folds
        print(f"🎯 CV folds: {args.cv_folds}")
    
    # Initialize and run pipeline
    try:
        print(f"\n🚀 Initializing Brain GNN Pipeline...")
        pipeline = BrainGNNPipeline(config)
        
        if args.model and args.model != 'all':
            print(f"🎯 Training single model: {args.model.upper()}")
            # TODO: Implement single model training
            print("⚠️  Single model training not yet implemented")
            print("    Use full pipeline for now")
        else:
            print(f"🔄 Running complete pipeline...")
            results = pipeline.run_complete_pipeline()
            
            # Print summary
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📊 Results saved to: {pipeline.results_path}")
            print(f"📈 Visualizations: {pipeline.plots_path}")
            print(f"🔍 Explainability: {pipeline.explainability_path}")
            
            if 'model_comparison' in results and results['model_comparison'].get('best_model'):
                best_model = results['model_comparison']['best_model']
                print(f"🏆 Best model: {best_model}")
    
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())