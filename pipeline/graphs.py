#!/usr/bin/env python3

import typer
from pathlib import Path
from .build_graphs import app as build_graphs_app
from .graph_analysis import run_complete_analysis
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

# Add the build_graphs commands to the main app
app.add_typer(build_graphs_app, name="build")

@app.command()
def analyze(
    graph_dir: Path = typer.Option(
        Path("derivatives/graphs"),
        help="Directory containing graph files",
        exists=True,
    ),
    output_dir: Path = typer.Option(
        Path("derivatives/graph_analysis"),
        help="Output directory for analysis results",
    ),
    subject_pattern: str = typer.Option(
        "*",
        help="Pattern to match subject IDs (e.g., 'YLOPDHC*')",
    ),
):
    """
    Analyze constructed brain graphs and create visualizations.
    """
    typer.echo(f"Analyzing graphs from {graph_dir}")
    typer.echo(f"Results will be saved to {output_dir}")
    
    try:
        run_complete_analysis(graph_dir, output_dir, subject_pattern)
        typer.echo("✓ Graph analysis completed successfully!")
    except Exception as e:
        typer.echo(f"✗ Graph analysis failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def pipeline(
    derivatives_dir: Path = typer.Option(
        Path("derivatives"),
        help="Path to the derivatives directory containing fMRIPrep output",
        exists=True,
    ),
    subject_pattern: str = typer.Option(
        "YLOPDHC*",
        help="Pattern to match subject IDs (e.g., 'YLOPDHC*')",
    ),
    n_rois: int = typer.Option(
        200,
        help="Number of ROIs in Schaefer atlas",
    ),
    correlation_threshold: float = typer.Option(
        0.3,
        help="Threshold for functional connectivity",
    ),
    max_workers: int = typer.Option(
        4,
        help="Maximum number of parallel processes",
    ),
):
    """
    Run the complete pipeline: build graphs and analyze them.
    """
    
    graph_dir = derivatives_dir / "graphs"
    analysis_dir = derivatives_dir / "graph_analysis"
    
    typer.echo("🧠 Running complete multimodal brain graph pipeline")
    typer.echo(f"Input: {derivatives_dir}")
    typer.echo(f"Subjects: {subject_pattern}")
    typer.echo(f"Atlas: Schaefer {n_rois} ROIs")
    typer.echo(f"Correlation threshold: {correlation_threshold}")
    
    # Step 1: Build graphs
    typer.echo("\n📊 Step 1: Building multimodal graphs...")
    
    try:
        from .build_graphs import build_graphs as build_graphs_func
        
        # Call the build_graphs function directly
        ctx = typer.Context(build_graphs_func)
        ctx.invoke(
            build_graphs_func,
            derivatives_dir=derivatives_dir,
            output_dir=graph_dir,
            subject_pattern=subject_pattern,
            n_rois=n_rois,
            correlation_threshold=correlation_threshold,
            max_workers=max_workers,
            normalize_features=True
        )
        
        typer.echo("✓ Graph construction completed!")
        
    except Exception as e:
        typer.echo(f"✗ Graph construction failed: {e}", err=True)
        raise typer.Exit(1)
    
    # Step 2: Analyze graphs
    typer.echo("\n📈 Step 2: Analyzing graphs...")
    
    try:
        run_complete_analysis(graph_dir, analysis_dir, subject_pattern)
        typer.echo("✓ Graph analysis completed!")
        
    except Exception as e:
        typer.echo(f"✗ Graph analysis failed: {e}", err=True)
        raise typer.Exit(1)
    
    typer.echo("\n🎉 Complete pipeline finished successfully!")
    typer.echo(f"📁 Graph files: {graph_dir}")
    typer.echo(f"📁 Analysis results: {analysis_dir}")
    typer.echo(f"📁 Plots: {analysis_dir / 'plots'}")

@app.command()
def info():
    """
    Display information about the multimodal brain graph construction pipeline.
    """
    info_text = """
🧠 Multimodal Brain Graph Construction Pipeline

This pipeline constructs multimodal brain graphs from preprocessed fMRI and structural MRI data.

OVERVIEW:
---------
1. Functional Graph Construction
   • Nodes: 177+ ROIs from Schaefer atlas
   • Edges: Thresholded Pearson correlations between ROI time series
   • Features: Graph-theoretic metrics (degree, betweenness, clustering, etc.)

2. Structural Graph Construction  
   • Nodes: Same ROIs as functional graph
   • Features: Regional morphometrics (thickness, area, volume, curvature)
   • Edges: Morphometric similarity between regions

3. Multimodal Integration
   • Node Features: Combined functional + structural descriptors
   • Graph Representation: Unified multimodal brain networks

REQUIRED DATA:
--------------
• Preprocessed BOLD data (fMRIPrep output in MNI space)
• FreeSurfer morphometric data (cortical and subcortical measures)
• Data should be organized in BIDS derivatives format

OUTPUTS:
--------
• Individual subject graphs (.pkl files)
• Network metrics (CSV)
• Hub analysis (JSON)
• Consensus group networks
• Visualization plots

USAGE EXAMPLES:
---------------
# Build graphs for all subjects
python -m pipeline.graphs build build-graphs --subject-pattern "YLOPDHC*"

# Analyze existing graphs
python -m pipeline.graphs analyze --graph-dir derivatives/graphs

# Run complete pipeline
python -m pipeline.graphs pipeline --subject-pattern "YLOPDHC*"

For more help: python -m pipeline.graphs [command] --help
    """
    
    typer.echo(info_text)

if __name__ == "__main__":
    app()