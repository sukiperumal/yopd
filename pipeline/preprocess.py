#!/usr/bin/env python3

import typer
import subprocess
from pathlib import Path
import os
import time

app = typer.Typer()

def run_fmriprep_docker(bids_dir: Path, output_dir: Path, participant_label: str):
    """Run fMRIPrep-docker for a single subject"""
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the absolute paths
    bids_dir = bids_dir.absolute()
    output_dir = output_dir.absolute()
    
    # Create the Docker command
    # Ensure the license file exists
    license_file = Path('licenses/license.txt').absolute()
    if not license_file.exists():
        raise typer.BadParameter(f"FreeSurfer license file not found at {license_file}")
    
    cmd = [
        'docker', 'run', '--rm',
        '--platform', 'linux/amd64',
        '--security-opt', 'seccomp=unconfined',
        # Add memory limit
        '--memory', '8g',
        '--memory-swap', '8g',
        # Add CPU limit
        '--cpus', '4',
        '-e', 'DOCKER_DEFAULT_PLATFORM=linux/amd64',
        '--privileged',
        '-v', f'{bids_dir}:/data:ro,delegated',
        '-v', f'{output_dir}:/out:rw,delegated',
        '-v', f'{str(Path.home())}/.cache/fmriprep:/work:rw,delegated',
        '-v', f'{license_file}:/opt/freesurfer/license.txt:ro',
        'nipreps/fmriprep:25.2.3',  # Update to latest version
        '/data', '/out', 'participant',
        '--participant-label', participant_label,
        '--fs-license-file', '/opt/freesurfer/license.txt',
        '--output-spaces', 'MNI152NLin2009cAsym',
        # Reduce number of processes
        '--nthreads', '4',
        '--omp-nthreads', '2',
        # Add memory-saving options
        '--low-mem',
        '--mem-mb', '8192',
        '--skip-bids-validation',
    ]
    
    try:
        subprocess.run(cmd, check=True)
        typer.echo(f"✓ fMRIPrep processing completed successfully for sub-{participant_label}")
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running fMRIPrep: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def batch(
    bids_dir: Path = typer.Option(
        Path("data"),
        help="Path to the BIDS directory containing the data",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("/Volumes/Extreme SSD/data_NIMHANS/derivatives/fmriprep"),
        help="Path where derivatives will be stored",
    ),
    pattern: str = typer.Option(
        "YLOPDHC*",
        help="Pattern to match subject IDs (e.g., 'YLOPDHC*')",
    ),
):
    """
    Run fMRIPrep-docker on multiple participants matching a pattern.
    """
    # Get all subject directories matching the pattern
    bids_dir = Path(bids_dir)
    subjects = []
    
    for subject_dir in bids_dir.glob(f"sub-{pattern}"):
        subject_id = subject_dir.name.replace('sub-', '')
        subjects.append(subject_id)
    
    if not subjects:
        typer.echo(f"No subjects found matching pattern: {pattern}")
        raise typer.Exit(1)
    
    # Sort subjects for consistent processing order
    subjects.sort()
    
    typer.echo(f"Found {len(subjects)} subjects to process:")
    for subject in subjects:
        typer.echo(f"- {subject}")
    
    # Process each subject with error handling and delays
    for i, subject in enumerate(subjects):
        typer.echo(f"\nProcessing participant {i+1}/{len(subjects)}: sub-{subject}")
        typer.echo(f"BIDS directory: {bids_dir}")
        typer.echo(f"Output directory: {output_dir}")
        
        try:
            run_fmriprep_docker(bids_dir, output_dir, subject)
            
            # Clean up between subjects
            subprocess.run(['docker', 'system', 'prune', '-f'], check=False)
            
            # Add delay between subjects
            if i < len(subjects) - 1:
                typer.echo("Waiting 60 seconds before next subject...")
                time.sleep(60)
                
        except Exception as e:
            typer.echo(f"Error processing subject {subject}: {e}", err=True)
            continue

@app.command()
def main(
    participant_label: str = typer.Argument(..., help="Participant label (without 'sub-' prefix)"),
    bids_dir: Path = typer.Option(
        Path("data"),
        help="Path to the BIDS directory containing the data",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("derivatives/fmriprep"),
        help="Path where derivatives will be stored",
    ),
):
    """
    Run fMRIPrep-docker on a single participant's data.
    """
    # Remove 'sub-' prefix if present
    participant_label = participant_label.replace('sub-', '')
    
    typer.echo(f"Processing participant: sub-{participant_label}")
    typer.echo(f"BIDS directory: {bids_dir}")
    typer.echo(f"Output directory: {output_dir}")
    
    run_fmriprep_docker(bids_dir, output_dir, participant_label)

if __name__ == "__main__":
    app()