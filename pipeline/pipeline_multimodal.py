#!/usr/bin/env python3

"""
Unified Multimodal Brain Graph Pipeline

This script provides a unified interface for the three-stage brain graph construction
pipeline: structural (T1w), functional (fMRI), and multimodal integration.

Usage:
    python pipeline_multimodal.py <command> <subject_dir> [options]

Commands:
    structural  - Build structural graph from T1w data
    functional  - Build functional graph from fMRI data  
    multimodal  - Integrate functional and structural graphs
    complete    - Run all three stages sequentially

Author: Pipeline Development Team
Date: November 2025
"""

import typer
from pathlib import Path
from typing import Optional
import logging
import json
import sys

# Add the pipeline modules to path
sys.path.append(str(Path(__file__).parent))

from graph_t1w import construct_structural_graph_cli, StructuralConfig, StructuralGraphConstructor
from graph_fmri import construct_functional_graph_cli, FunctionalConfig, FunctionalGraphConstructor  
from graph_merge import integrate_multimodal_graphs_cli, MultimodalConfig, MultimodalGraphIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def structural(
    subject_dir: Path = typer.Argument(..., help="Path to fMRIPrep subject directory (e.g., derivatives/fmriprep/sub-01)"),
    output_file: Optional[str] = typer.Option(None, help="Output file path"),
    n_rois: int = typer.Option(200, help="Number of ROIs in Schaefer atlas (100, 200, 400, 500, 600, 800, 1000)"),
    similarity_threshold: float = typer.Option(0.1, help="Threshold for structural connectivity (0.0-1.0)"),
    normalize_features: bool = typer.Option(True, help="Whether to normalize morphometric features"),
    use_surface_features: bool = typer.Option(True, help="Include surface-based features (thickness, area, curvature)"),
    use_volume_features: bool = typer.Option(True, help="Include volume-based features (GM/WM volumes)")
):
    """Build structural graph from T1w MRI data processed by fMRIPrep."""
    
    try:
        typer.echo(f"🧠 Building structural graph for: {subject_dir.name}")
        typer.echo(f"   Atlas: Schaefer {n_rois} ROIs")
        typer.echo(f"   Similarity threshold: {similarity_threshold}")
        typer.echo(f"   Normalize features: {normalize_features}")
        
        # Configure and run structural graph construction
        config = StructuralConfig(
            schaefer_n_rois=n_rois,
            similarity_threshold=similarity_threshold,
            normalize_features=normalize_features,
            use_surface_features=use_surface_features,
            use_volume_features=use_volume_features
        )
        
        constructor = StructuralGraphConstructor(config)
        structural_graph = constructor.construct_structural_graph(subject_dir)
        
        # Determine output file
        if output_file is None:
            output_file = f"{subject_dir.name}_structural_graph.pkl"
        
        # Save graph
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(structural_graph, f)
        
        typer.echo(f"✅ Structural graph saved to: {output_file}")
        
        # Print summary
        n_edges = (structural_graph['similarity_matrix'] > similarity_threshold).sum() // 2
        typer.echo(f"📊 Summary:")
        typer.echo(f"   - ROIs: {structural_graph['n_rois']}")
        typer.echo(f"   - Edges: {n_edges}")
        typer.echo(f"   - Features per ROI: {structural_graph['roi_features'].shape[1]}")
        
    except Exception as e:
        typer.echo(f"❌ Structural graph construction failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def functional(
    subject_dir: Path = typer.Argument(..., help="Path to fMRIPrep subject directory"),
    output_file: Optional[str] = typer.Option(None, help="Output file path"),
    task: Optional[str] = typer.Option(None, help="Task name (e.g., 'rest', 'task')"),
    session: Optional[str] = typer.Option(None, help="Session identifier (e.g., '01')"),
    run: Optional[str] = typer.Option(None, help="Run identifier (e.g., '1')"),
    n_rois: int = typer.Option(200, help="Number of ROIs in Schaefer atlas"),
    correlation_threshold: float = typer.Option(0.3, help="Threshold for functional connectivity (0.0-1.0)"),
    connectivity_metric: str = typer.Option('correlation', help="Connectivity measure ('correlation', 'partial correlation')"),
    high_pass: Optional[float] = typer.Option(0.01, help="High-pass filter cutoff (Hz)"),
    low_pass: Optional[float] = typer.Option(0.1, help="Low-pass filter cutoff (Hz)")
):
    """Build functional graph from fMRI data processed by fMRIPrep."""
    
    try:
        typer.echo(f"📡 Building functional graph for: {subject_dir.name}")
        typer.echo(f"   Atlas: Schaefer {n_rois} ROIs")
        typer.echo(f"   Correlation threshold: {correlation_threshold}")
        typer.echo(f"   Connectivity metric: {connectivity_metric}")
        if task:
            typer.echo(f"   Task: {task}")
        if session:
            typer.echo(f"   Session: {session}")
        if run:
            typer.echo(f"   Run: {run}")
        
        # Configure and run functional graph construction
        config = FunctionalConfig(
            schaefer_n_rois=n_rois,
            correlation_threshold=correlation_threshold,
            connectivity_metric=connectivity_metric,
            high_pass=high_pass,
            low_pass=low_pass
        )
        
        constructor = FunctionalGraphConstructor(config)
        functional_graph = constructor.construct_functional_graph(
            subject_dir, task=task, session=session, run=run
        )
        
        # Determine output file
        if output_file is None:
            output_file = f"{subject_dir.name}_functional_graph.pkl"
        
        # Save graph
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(functional_graph, f)
        
        typer.echo(f"✅ Functional graph saved to: {output_file}")
        
        # Print summary
        n_edges = (functional_graph['adjacency_matrix'] != 0).sum() // 2
        typer.echo(f"📊 Summary:")
        typer.echo(f"   - ROIs: {functional_graph['n_rois']}")
        typer.echo(f"   - Edges: {n_edges}")
        typer.echo(f"   - Timepoints: {functional_graph['n_timepoints']}")
        typer.echo(f"   - TR: {functional_graph.get('tr', 'unknown')} seconds")
        
    except Exception as e:
        typer.echo(f"❌ Functional graph construction failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def multimodal(
    functional_graph: Path = typer.Argument(..., help="Path to functional graph pickle file"),
    structural_graph: Path = typer.Argument(..., help="Path to structural graph pickle file"),
    output_file: Optional[str] = typer.Option(None, help="Output file path"),
    functional_weight: float = typer.Option(0.6, help="Weight for functional modality (0.0-1.0)"),
    structural_weight: float = typer.Option(0.4, help="Weight for structural modality (0.0-1.0)"),
    integration_method: str = typer.Option('concatenate', help="Feature integration method ('concatenate', 'weighted_average', 'pca')"),
    edge_integration_method: str = typer.Option('weighted_sum', help="Edge integration method ('weighted_sum', 'product', 'max')"),
    normalize_before_integration: bool = typer.Option(True, help="Normalize features before integration")
):
    """Integrate functional and structural graphs into multimodal representation."""
    
    try:
        typer.echo(f"🔗 Integrating multimodal graphs")
        typer.echo(f"   Functional: {functional_graph.name}")
        typer.echo(f"   Structural: {structural_graph.name}")
        typer.echo(f"   Functional weight: {functional_weight}")
        typer.echo(f"   Structural weight: {structural_weight}")
        typer.echo(f"   Integration method: {integration_method}")
        
        # Load graphs
        import pickle
        
        with open(functional_graph, 'rb') as f:
            func_graph = pickle.load(f)
        
        with open(structural_graph, 'rb') as f:
            struct_graph = pickle.load(f)
        
        # Configure and run integration
        config = MultimodalConfig(
            functional_weight=functional_weight,
            structural_weight=structural_weight,
            integration_method=integration_method,
            edge_integration_method=edge_integration_method,
            normalize_before_integration=normalize_before_integration
        )
        
        integrator = MultimodalGraphIntegrator(config)
        multimodal_result = integrator.integrate_graphs(func_graph, struct_graph)
        
        # Determine output file
        if output_file is None:
            base_name = functional_graph.stem.replace('_functional_graph', '')
            output_file = f"{base_name}_multimodal_graph.pkl"
        
        # Save result
        with open(output_file, 'wb') as f:
            pickle.dump(multimodal_result, f)
        
        typer.echo(f"✅ Multimodal graph saved to: {output_file}")
        
        # Print summary
        n_edges = (multimodal_result['multimodal_adjacency'] != 0).sum() // 2
        typer.echo(f"📊 Summary:")
        typer.echo(f"   - ROIs: {multimodal_result['n_rois']}")
        typer.echo(f"   - Multimodal edges: {n_edges}")
        typer.echo(f"   - Features per ROI: {multimodal_result['multimodal_features'].shape[1]}")
        
    except Exception as e:
        typer.echo(f"❌ Multimodal integration failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def complete(
    subject_dir: Path = typer.Argument(..., help="Path to fMRIPrep subject directory"),
    output_dir: Optional[Path] = typer.Option(None, help="Output directory for graph files"),
    task: Optional[str] = typer.Option(None, help="Task name for functional data"),
    session: Optional[str] = typer.Option(None, help="Session identifier"),
    run: Optional[str] = typer.Option(None, help="Run identifier"),
    n_rois: int = typer.Option(200, help="Number of ROIs in Schaefer atlas"),
    correlation_threshold: float = typer.Option(0.3, help="Functional connectivity threshold"),
    similarity_threshold: float = typer.Option(0.1, help="Structural similarity threshold"),
    functional_weight: float = typer.Option(0.6, help="Weight for functional modality in integration"),
    structural_weight: float = typer.Option(0.4, help="Weight for structural modality in integration")
):
    """Run complete pipeline: structural → functional → multimodal integration."""
    
    try:
        # Setup output directory
        if output_dir is None:
            output_dir = Path("output") / subject_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        typer.echo(f"🚀 Running complete multimodal pipeline for: {subject_dir.name}")
        typer.echo(f"📁 Output directory: {output_dir}")
        typer.echo()
        
        # Step 1: Structural graph
        typer.echo("1️⃣ Building structural graph...")
        struct_config = StructuralConfig(
            schaefer_n_rois=n_rois,
            similarity_threshold=similarity_threshold,
            normalize_features=True
        )
        
        struct_constructor = StructuralGraphConstructor(struct_config)
        structural_graph = struct_constructor.construct_structural_graph(subject_dir)
        
        struct_file = output_dir / f"{subject_dir.name}_structural_graph.pkl"
        import pickle
        with open(struct_file, 'wb') as f:
            pickle.dump(structural_graph, f)
        
        typer.echo(f"   ✅ Structural graph saved: {struct_file.name}")
        
        # Step 2: Functional graph
        typer.echo("2️⃣ Building functional graph...")
        func_config = FunctionalConfig(
            schaefer_n_rois=n_rois,
            correlation_threshold=correlation_threshold,
            connectivity_metric='correlation'
        )
        
        func_constructor = FunctionalGraphConstructor(func_config)
        functional_graph = func_constructor.construct_functional_graph(
            subject_dir, task=task, session=session, run=run
        )
        
        func_file = output_dir / f"{subject_dir.name}_functional_graph.pkl"
        with open(func_file, 'wb') as f:
            pickle.dump(functional_graph, f)
        
        typer.echo(f"   ✅ Functional graph saved: {func_file.name}")
        
        # Step 3: Multimodal integration
        typer.echo("3️⃣ Integrating multimodal graph...")
        multimodal_config = MultimodalConfig(
            functional_weight=functional_weight,
            structural_weight=structural_weight,
            integration_method='concatenate',
            normalize_before_integration=True
        )
        
        integrator = MultimodalGraphIntegrator(multimodal_config)
        multimodal_result = integrator.integrate_graphs(functional_graph, structural_graph)
        
        multimodal_file = output_dir / f"{subject_dir.name}_multimodal_graph.pkl"
        with open(multimodal_file, 'wb') as f:
            pickle.dump(multimodal_result, f)
        
        typer.echo(f"   ✅ Multimodal graph saved: {multimodal_file.name}")
        
        # Create summary
        summary = {
            'subject': subject_dir.name,
            'n_rois': n_rois,
            'structural': {
                'file': struct_file.name,
                'n_edges': int((structural_graph['similarity_matrix'] > similarity_threshold).sum() // 2),
                'n_features': structural_graph['roi_features'].shape[1],
                'similarity_threshold': similarity_threshold
            },
            'functional': {
                'file': func_file.name,
                'n_edges': int((functional_graph['adjacency_matrix'] != 0).sum() // 2),
                'n_timepoints': functional_graph['n_timepoints'],
                'correlation_threshold': correlation_threshold,
                'task': task,
                'session': session,
                'run': run
            },
            'multimodal': {
                'file': multimodal_file.name,
                'n_edges': int((multimodal_result['multimodal_adjacency'] != 0).sum() // 2),
                'n_features': multimodal_result['multimodal_features'].shape[1],
                'functional_weight': functional_weight,
                'structural_weight': structural_weight
            }
        }
        
        summary_file = output_dir / f"{subject_dir.name}_pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        typer.echo()
        typer.echo("🎉 Complete pipeline finished successfully!")
        typer.echo(f"📊 Pipeline Summary:")
        typer.echo(f"   - Subject: {subject_dir.name}")
        typer.echo(f"   - Atlas: Schaefer {n_rois} ROIs")
        typer.echo(f"   - Structural edges: {summary['structural']['n_edges']}")
        typer.echo(f"   - Functional edges: {summary['functional']['n_edges']}")
        typer.echo(f"   - Multimodal edges: {summary['multimodal']['n_edges']}")
        typer.echo(f"   - Multimodal features: {summary['multimodal']['n_features']}")
        typer.echo()
        typer.echo(f"📁 Output files:")
        typer.echo(f"   - {struct_file}")
        typer.echo(f"   - {func_file}")
        typer.echo(f"   - {multimodal_file}")
        typer.echo(f"   - {summary_file}")
        
    except Exception as e:
        typer.echo(f"❌ Complete pipeline failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def info():
    """Display information about the multimodal brain graph pipeline."""
    
    info_text = """
🧠 Multimodal Brain Graph Construction Pipeline

This pipeline constructs brain graphs from neuroimaging data in three stages:

STAGE 1: STRUCTURAL GRAPH (graph_t1w.py)
─────────────────────────────────────────
• Input: T1w MRI data processed by fMRIPrep + FreeSurfer output
• Nodes: Brain regions from Schaefer atlas (100-1000 ROIs)
• Node Features: Morphometric measures (thickness, area, volume, curvature)
• Edges: Morphometric similarity between regions
• Output: Structural brain graph with morphological features

STAGE 2: FUNCTIONAL GRAPH (graph_fmri.py)  
──────────────────────────────────────────
• Input: fMRI data processed by fMRIPrep (preprocessed BOLD)
• Nodes: Same brain regions as structural graph
• Node Features: Graph metrics (degree, betweenness, clustering, etc.)
• Edges: Functional connectivity (correlation-based)
• Output: Functional brain graph with connectivity patterns

STAGE 3: MULTIMODAL INTEGRATION (graph_merge.py)
─────────────────────────────────────────────────
• Input: Structural + functional graphs from stages 1 & 2
• Integration: Combines node features and connectivity patterns
• Methods: Feature concatenation, weighted averaging, PCA
• Output: Unified multimodal brain graph

PIPELINE COMMANDS:
─────────────────
• structural  - Build structural graph from T1w data
• functional  - Build functional graph from fMRI data
• multimodal  - Integrate structural + functional graphs  
• complete    - Run all three stages automatically

EXAMPLE USAGE:
─────────────
# Run complete pipeline
python pipeline_multimodal.py complete derivatives/fmriprep/sub-01

# Run individual stages
python pipeline_multimodal.py structural derivatives/fmriprep/sub-01
python pipeline_multimodal.py functional derivatives/fmriprep/sub-01 --task rest
python pipeline_multimodal.py multimodal sub-01_functional_graph.pkl sub-01_structural_graph.pkl

REQUIREMENTS:
────────────
• fMRIPrep processed data (anatomical + functional)
• FreeSurfer recon-all output
• Python packages: numpy, pandas, nibabel, nilearn, networkx, scipy, sklearn

For detailed help: python pipeline_multimodal.py <command> --help
    """
    
    typer.echo(info_text)

if __name__ == "__main__":
    app()