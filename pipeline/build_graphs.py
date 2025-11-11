#!/usr/bin/env python3

import typer
import logging
from pathlib import Path
from typing import List, Optional, Dict
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from .graph_construction import BrainGraphBuilder, GraphConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

def find_subject_data(derivatives_dir: Path, subject_id: str) -> Dict[str, Path]:
    """Find all required data files for a subject"""
    subject_data = {}
    
    # Look for fMRIPrep output
    fmriprep_dir = derivatives_dir / "fmriprep" / f"sub-{subject_id}"
    
    if fmriprep_dir.exists():
        # Find preprocessed BOLD file
        bold_pattern = f"*_task-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii*"
        bold_files = list(fmriprep_dir.rglob(bold_pattern))
        
        if bold_files:
            # Use the first BOLD file found (could be refined to select specific task)
            subject_data['bold_file'] = bold_files[0]
            logger.info(f"Found BOLD file: {bold_files[0].name}")
        else:
            logger.warning(f"No preprocessed BOLD files found for {subject_id}")
        
        # Find FreeSurfer directory
        freesurfer_dir = fmriprep_dir.parent / "sourcedata" / "freesurfer" / f"sub-{subject_id}"
        if freesurfer_dir.exists():
            subject_data['freesurfer_dir'] = freesurfer_dir
            logger.info(f"Found FreeSurfer directory: {freesurfer_dir}")
        else:
            logger.warning(f"No FreeSurfer directory found for {subject_id}")
    
    return subject_data

def process_single_subject(args) -> tuple:
    """Process a single subject - wrapper for multiprocessing"""
    subject_id, derivatives_dir, output_dir, config_dict = args
    
    try:
        logger.info(f"Processing subject: {subject_id}")
        
        # Create config from dictionary
        config = GraphConfig(**config_dict)
        
        # Find subject data
        subject_data = find_subject_data(Path(derivatives_dir), subject_id)
        
        if not subject_data:
            raise ValueError(f"No valid data found for subject {subject_id}")
        
        # Check required files
        required_files = ['bold_file', 'freesurfer_dir']
        missing_files = [f for f in required_files if f not in subject_data]
        
        if missing_files:
            raise ValueError(f"Missing required files for {subject_id}: {missing_files}")
        
        # Build graph
        graph_builder = BrainGraphBuilder(config)
        complete_graph = graph_builder.build_subject_graph(subject_data)
        
        # Save graph
        output_file = Path(output_dir) / f"sub-{subject_id}_multimodal_graph.pkl"
        graph_builder.save_graph(complete_graph, output_file)
        
        # Also save a summary as JSON
        summary_file = Path(output_dir) / f"sub-{subject_id}_graph_summary.json"
        save_graph_summary(complete_graph, summary_file)
        
        logger.info(f"Successfully processed subject {subject_id}")
        return subject_id, True, None
        
    except Exception as e:
        error_msg = f"Failed to process {subject_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return subject_id, False, error_msg

def save_graph_summary(graph_data: Dict, output_file: Path):
    """Save a summary of the graph data as JSON"""
    try:
        # Extract key metrics from the graph
        multimodal = graph_data['multimodal']
        functional = graph_data['functional']
        structural = graph_data['structural']
        
        summary = {
            'n_rois': multimodal['n_rois'],
            'functional_metrics': {
                'n_edges': int((functional['adjacency_matrix'] > 0).sum() / 2),
                'mean_correlation': float(functional['correlation_matrix'][functional['correlation_matrix'] > 0].mean()),
                'threshold': float(graph_data['config'].correlation_threshold)
            },
            'structural_metrics': {
                'n_features': structural['roi_features'].shape[1],
                'feature_names': structural['feature_names'],
                'mean_structural_connectivity': float(structural['structural_connectivity'].mean())
            },
            'multimodal_metrics': {
                'n_multimodal_features': multimodal['multimodal_features'].shape[1],
                'feature_names': multimodal['feature_names'],
                'n_multimodal_edges': int((multimodal['multimodal_adjacency'] > 0).sum() / 2)
            },
            'processing_info': {
                'atlas': 'Schaefer2018',
                'n_parcels': multimodal['n_rois'],
                'space': 'MNI152NLin2009cAsym'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save graph summary: {e}")

@app.command()
def build_graphs(
    derivatives_dir: Path = typer.Option(
        Path("derivatives"),
        help="Path to the derivatives directory containing fMRIPrep output",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("derivatives/graphs"),
        help="Path where graph files will be saved",
    ),
    subject_pattern: str = typer.Option(
        "YLOPDHC*",
        help="Pattern to match subject IDs (e.g., 'YLOPDHC*')",
    ),
    n_rois: int = typer.Option(
        200,
        help="Number of ROIs in Schaefer atlas (100, 200, 400, 500, 600, 800, 1000)",
    ),
    correlation_threshold: float = typer.Option(
        0.3,
        help="Threshold for functional connectivity (0.0 to 1.0)",
    ),
    max_workers: int = typer.Option(
        4,
        help="Maximum number of parallel processes",
    ),
    normalize_features: bool = typer.Option(
        True,
        help="Whether to normalize features",
    ),
):
    """
    Build multimodal brain graphs for multiple subjects.
    """
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subjects
    fmriprep_dir = derivatives_dir / "fmriprep"
    
    if not fmriprep_dir.exists():
        typer.echo(f"fMRIPrep directory not found: {fmriprep_dir}")
        raise typer.Exit(1)
    
    subjects = []
    for subject_dir in fmriprep_dir.glob(f"sub-{subject_pattern}"):
        subject_id = subject_dir.name.replace('sub-', '')
        subjects.append(subject_id)
    
    if not subjects:
        typer.echo(f"No subjects found matching pattern: {subject_pattern}")
        raise typer.Exit(1)
    
    subjects.sort()
    typer.echo(f"Found {len(subjects)} subjects to process: {subjects}")
    
    # Create configuration
    config_dict = {
        'schaefer_n_rois': n_rois,
        'correlation_threshold': correlation_threshold,
        'normalize_features': normalize_features,
        'output_format': 'networkx'
    }
    
    # Prepare arguments for parallel processing
    process_args = [
        (subject_id, str(derivatives_dir), str(output_dir), config_dict)
        for subject_id in subjects
    ]
    
    # Process subjects
    successful = []
    failed = []
    
    start_time = time.time()
    
    if max_workers == 1:
        # Sequential processing for debugging
        for args in process_args:
            subject_id, success, error = process_single_subject(args)
            if success:
                successful.append(subject_id)
            else:
                failed.append((subject_id, error))
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_subject = {
                executor.submit(process_single_subject, args): args[0]
                for args in process_args
            }
            
            # Collect results
            for future in as_completed(future_to_subject):
                subject_id, success, error = future.result()
                if success:
                    successful.append(subject_id)
                else:
                    failed.append((subject_id, error))
    
    # Report results
    elapsed_time = time.time() - start_time
    typer.echo(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    typer.echo(f"✓ Successfully processed: {len(successful)} subjects")
    
    if successful:
        typer.echo(f"Successful subjects: {', '.join(successful)}")
    
    if failed:
        typer.echo(f"✗ Failed to process: {len(failed)} subjects")
        for subject_id, error in failed:
            typer.echo(f"  - {subject_id}: {error}")
    
    # Create overall summary
    create_cohort_summary(successful, output_dir, config_dict)

def create_cohort_summary(successful_subjects: List[str], output_dir: Path, config_dict: Dict):
    """Create a summary for the entire cohort"""
    try:
        cohort_summary = {
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_subjects': len(successful_subjects),
            'subjects': successful_subjects,
            'config': config_dict,
            'output_files': {
                'graph_files': [f"sub-{sub}_multimodal_graph.pkl" for sub in successful_subjects],
                'summary_files': [f"sub-{sub}_graph_summary.json" for sub in successful_subjects]
            }
        }
        
        summary_file = output_dir / "cohort_graph_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(cohort_summary, f, indent=2)
        
        logger.info(f"Created cohort summary: {summary_file}")
        
    except Exception as e:
        logger.error(f"Failed to create cohort summary: {e}")

@app.command()
def analyze_graphs(
    graph_dir: Path = typer.Option(
        Path("derivatives/graphs"),
        help="Directory containing graph files",
        exists=True,
    ),
    output_file: Path = typer.Option(
        Path("derivatives/graphs/graph_analysis.json"),
        help="Output file for analysis results",
    ),
):
    """
    Analyze constructed graphs and extract group-level statistics.
    """
    
    # Find all graph files
    graph_files = list(graph_dir.glob("*_multimodal_graph.pkl"))
    
    if not graph_files:
        typer.echo(f"No graph files found in {graph_dir}")
        raise typer.Exit(1)
    
    typer.echo(f"Analyzing {len(graph_files)} graph files")
    
    # TODO: Implement graph analysis
    # This would include:
    # - Group-level network statistics
    # - Hub identification
    # - Small-world properties
    # - Modularity analysis
    # - Between-group comparisons (if applicable)
    
    typer.echo("Graph analysis functionality coming soon...")

if __name__ == "__main__":
    app()