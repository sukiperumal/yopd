#!/usr/bin/env python3
"""
Complete GNN Pipeline Runner for YOPD Dataset

This script:
1. Validates BIDS structure (basic check)
2. Builds multimodal brain graphs from preprocessed data
3. Runs the GNN classification pipeline
4. Outputs results to gnn_pipeline/outputs

Usage:
    python run_complete_gnn_pipeline.py --data-dir "/Volumes/Extreme SSD/data_NIMHANS - Copy/outputs/01_preprocessed/"
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pickle
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_bids_structure(data_dir: Path) -> tuple[bool, List[str]]:
    """
    Perform basic BIDS validation on the preprocessed data directory
    
    Returns:
        tuple: (is_valid, list_of_issues)
    """
    logger.info("=" * 80)
    logger.info("STEP 1: BIDS STRUCTURE VALIDATION")
    logger.info("=" * 80)
    
    issues = []
    
    # Check if directory exists
    if not data_dir.exists():
        issues.append(f"Data directory does not exist: {data_dir}")
        return False, issues
    
    logger.info(f"✓ Data directory exists: {data_dir}")
    
    # Find subject directories
    subject_dirs = sorted(list(data_dir.glob("sub-*")))
    
    if not subject_dirs:
        issues.append("No subject directories found (expected directories matching 'sub-*')")
        return False, issues
    
    logger.info(f"✓ Found {len(subject_dirs)} subject directories")
    
    # Check each subject directory for required files
    valid_subjects = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        
        # Look for T1w brain files (structural)
        t1w_files = list(subject_dir.glob("*T1w_brain.nii*"))
        
        # Look for masks
        brain_masks = list(subject_dir.glob("*brain_mask.nii*"))
        gm_masks = list(subject_dir.glob("*GM_mask.nii*"))
        wm_masks = list(subject_dir.glob("*WM_mask.nii*"))
        
        subject_valid = True
        subject_issues = []
        
        if not t1w_files:
            subject_issues.append(f"  {subject_id}: Missing T1w brain file")
            subject_valid = False
        
        if not brain_masks:
            subject_issues.append(f"  {subject_id}: Missing brain mask")
            subject_valid = False
        
        if subject_valid:
            valid_subjects.append(subject_dir)
            logger.info(f"  ✓ {subject_id}: Valid")
        else:
            for issue in subject_issues:
                logger.warning(issue)
                issues.append(issue)
    
    if valid_subjects:
        logger.info(f"\n✓ Found {len(valid_subjects)} valid subjects for processing")
        return True, issues
    else:
        issues.append("No valid subjects found with required preprocessed data")
        return False, issues


def create_dataset_description(data_dir: Path):
    """Create a basic dataset_description.json if it doesn't exist"""
    desc_file = data_dir / "dataset_description.json"
    
    if not desc_file.exists():
        logger.info("Creating dataset_description.json...")
        description = {
            "Name": "YOPD Preprocessed Dataset",
            "BIDSVersion": "1.6.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "YOPD Preprocessing Pipeline",
                    "Description": "Preprocessed structural MRI data"
                }
            ]
        }
        
        with open(desc_file, 'w') as f:
            json.dump(description, f, indent=2)
        
        logger.info(f"✓ Created dataset_description.json")


def build_graphs_from_preprocessed_data(data_dir: Path, output_dir: Path) -> List[Path]:
    """
    Build brain graphs from preprocessed data
    
    Note: This is a simplified version. The full pipeline would typically
    include functional connectivity from fMRI, but we'll create structural
    graphs from the available T1w data.
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: BUILD BRAIN GRAPHS FROM PREPROCESSED DATA")
    logger.info("=" * 80)
    
    # Find all subject directories
    subject_dirs = sorted(list(data_dir.glob("sub-*")))
    
    if not subject_dirs:
        logger.error("No subject directories found!")
        return []
    
    logger.info(f"Found {len(subject_dirs)} subjects to process")
    
    graph_files = []
    
    # For now, we'll create placeholder graphs since we need to properly implement
    # the graph construction from the preprocessed T1w data
    # In a full implementation, this would call graph_t1w.py or similar
    
    logger.warning("\n⚠️  Graph construction from preprocessed data requires:")
    logger.warning("    - Atlas parcellation (e.g., Schaefer, AAL)")
    logger.warning("    - ROI extraction and feature computation")
    logger.warning("    - Connectivity estimation")
    logger.warning("\n    For now, checking if graphs already exist...")
    
    # Check if graphs already exist in the output directory
    existing_graphs = list(output_dir.glob("*comprehensive_multimodal_graph.pkl"))
    
    if existing_graphs:
        logger.info(f"\n✓ Found {len(existing_graphs)} existing graph files:")
        for graph_file in existing_graphs:
            logger.info(f"  - {graph_file.name}")
        graph_files = existing_graphs
    else:
        logger.error("\n❌ No existing graph files found!")
        logger.error("Please run the graph construction pipeline first:")
        logger.error("  python pipeline/pipeline_multimodal.py complete <subject_dir>")
        logger.error("\nOr ensure graph .pkl files exist in the output directory.")
    
    return graph_files


def run_gnn_pipeline(graphs_dir: Path, output_dir: Path):
    """
    Run the GNN classification pipeline
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: RUN GNN CLASSIFICATION PIPELINE")
    logger.info("=" * 80)
    
    # Import the GNN pipeline
    gnn_pipeline_dir = Path(__file__).parent / "gnn_pipeline"
    sys.path.insert(0, str(gnn_pipeline_dir))
    sys.path.insert(0, str(gnn_pipeline_dir / "src"))
    sys.path.insert(0, str(gnn_pipeline_dir / "scripts"))
    
    try:
        from run_pipeline import BrainGNNPipeline, create_default_config
        
        # Create configuration
        config = create_default_config()
        
        # Update paths
        config['data']['comprehensive_graphs_dir'] = str(graphs_dir)
        config['output']['base_dir'] = str(output_dir)
        
        logger.info(f"Configuration:")
        logger.info(f"  Input graphs: {graphs_dir}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Training epochs: {config['training']['num_epochs']}")
        logger.info(f"  CV folds: {config['training']['cv_folds']}")
        
        # Initialize pipeline
        logger.info("\n🚀 Initializing Brain GNN Pipeline...")
        pipeline = BrainGNNPipeline(config)
        
        # Run complete pipeline
        logger.info("🔄 Running complete pipeline (this may take a while)...")
        results = pipeline.run_complete_pipeline()
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📊 Results saved to: {output_dir}")
        
        if 'model_comparison' in results and results['model_comparison'].get('best_model'):
            best_model = results['model_comparison']['best_model']
            logger.info(f"🏆 Best model: {best_model}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Complete GNN Pipeline for YOPD Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to preprocessed data directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/sukiperumal/Documents/yopd/gnn_pipeline/outputs',
        help='Output directory for GNN pipeline results'
    )
    
    parser.add_argument(
        '--graphs-dir',
        type=str,
        help='Directory containing brain graph .pkl files (if different from output-dir)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip BIDS validation step'
    )
    
    parser.add_argument(
        '--skip-graph-building',
        action='store_true',
        help='Skip graph building step (use existing graphs)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    graphs_dir = Path(args.graphs_dir) if args.graphs_dir else output_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("YOPD GNN PIPELINE EXECUTION")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Graphs directory: {graphs_dir}")
    logger.info("")
    
    try:
        # Step 1: Validate BIDS structure
        if not args.skip_validation:
            is_valid, issues = validate_bids_structure(data_dir)
            
            if not is_valid:
                logger.error("\n❌ BIDS validation failed with issues:")
                for issue in issues:
                    logger.error(f"  - {issue}")
                logger.error("\nUse --skip-validation to proceed anyway")
                return 1
            
            # Create dataset description if needed
            create_dataset_description(data_dir)
        else:
            logger.info("Skipping BIDS validation (--skip-validation)")
        
        # Step 2: Build graphs from preprocessed data
        if not args.skip_graph_building:
            graph_files = build_graphs_from_preprocessed_data(data_dir, graphs_dir)
            
            if not graph_files:
                logger.error("\n❌ No graph files available for GNN pipeline!")
                logger.error("Please build graphs first or use --graphs-dir to specify existing graphs")
                return 1
        else:
            logger.info("\nSkipping graph building (--skip-graph-building)")
            # Check if graphs exist
            graph_files = list(graphs_dir.glob("*comprehensive_multimodal_graph.pkl"))
            if not graph_files:
                logger.error(f"❌ No graph files found in {graphs_dir}")
                return 1
            logger.info(f"✓ Found {len(graph_files)} existing graph files")
        
        # Step 3: Run GNN pipeline
        results = run_gnn_pipeline(graphs_dir, output_dir)
        
        logger.info("\n✅ Complete pipeline executed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
