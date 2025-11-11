#!/usr/bin/env python3

import typer
import subprocess
from pathlib import Path
import os
import time
import concurrent.futures
import psutil

app = typer.Typer()

def get_available_resources():
    """Get available system resources for container allocation"""
    cpu_count = psutil.cpu_count(logical=True)
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # Convert to GB
    
    # Reserve some resources for the system
    available_cpus = max(cpu_count - 2, 1)
    available_memory = max(total_memory - 4, 4)  # Reserve 4GB for system
    
    return available_cpus, available_memory

def calculate_container_resources(total_containers):
    """Calculate resources per container based on system availability"""
    available_cpus, available_memory = get_available_resources()
    
    cpus_per_container = max(4, int(available_cpus / total_containers))
    memory_per_container = max(8, int(available_memory / total_containers))
    
    return cpus_per_container, memory_per_container

def run_fmriprep_docker(bids_dir: Path, output_dir: Path, participant_label: str, 
                       cpus: int = 4, memory: int = 8):
    """Run fMRIPrep-docker for a single subject with specified resources"""
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the absolute paths
    bids_dir = bids_dir.absolute()
    output_dir = output_dir.absolute()
    work_dir = output_dir / f"work_sub-{participant_label}"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure the license file exists
    license_file = Path('licenses/license.txt').absolute()
    if not license_file.exists():
        raise typer.BadParameter(f"FreeSurfer license file not found at {license_file}")
    
    cmd = [
        'docker', 'run', '--rm',
        '--platform', 'linux/amd64',
        '--security-opt', 'seccomp=unconfined',
        # Add memory limit
        '--memory', f'{memory}g',
        '--memory-swap', f'{memory}g',
        # Add CPU limit
        '--cpus', str(cpus),
        '-e', 'DOCKER_DEFAULT_PLATFORM=linux/amd64',
        '--privileged',
        '-v', f'{bids_dir}:/data:ro,delegated',
        '-v', f'{output_dir}:/out:rw,delegated',
        '-v', f'{work_dir}:/work:rw,delegated',
        '-v', f'{license_file}:/opt/freesurfer/license.txt:ro',
        f'--name=fmriprep_sub-{participant_label}',
        'nipreps/fmriprep:25.2.3',
        '/data', '/out', 'participant',
        '--participant-label', participant_label,
        '--fs-license-file', '/opt/freesurfer/license.txt',
        '--output-spaces', 'MNI152NLin2009cAsym',
        '--nthreads', str(cpus),
        '--omp-nthreads', str(max(2, cpus // 2)),
        '--low-mem',
        '--mem-mb', str(memory * 1024),
        '--skip-bids-validation',
    ]
    
    try:
        subprocess.run(cmd, check=True)
        typer.echo(f"✓ fMRIPrep processing completed successfully for sub-{participant_label}")
        return True
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error running fMRIPrep for sub-{participant_label}: {e}", err=True)
        return False

def process_subject(args):
    """Wrapper function for parallel processing"""
    bids_dir, output_dir, subject, cpus, memory = args
    return run_fmriprep_docker(bids_dir, output_dir, subject, cpus, memory)

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
    max_parallel: int = typer.Option(
        4,
        help="Maximum number of subjects to process in parallel",
    ),
):
    """
    Run fMRIPrep-docker on multiple participants in parallel.
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
    
    subjects.sort()
    
    # Calculate resources per container
    cpus_per_container, memory_per_container = calculate_container_resources(max_parallel)
    
    typer.echo(f"Found {len(subjects)} subjects to process")
    typer.echo(f"Processing up to {max_parallel} subjects in parallel")
    typer.echo(f"Resources per container: {cpus_per_container} CPUs, {memory_per_container}GB RAM")
    
    # Prepare arguments for parallel processing
    process_args = [
        (bids_dir, output_dir, subject, cpus_per_container, memory_per_container)
        for subject in subjects
    ]
    
    # Process subjects in parallel using a thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = list(executor.map(process_subject, process_args))
    
    # Report results
    successful = sum(1 for result in futures if result)
    failed = len(subjects) - successful
    
    typer.echo(f"\nProcessing complete:")
    typer.echo(f"✓ Successfully processed: {successful} subjects")
    if failed > 0:
        typer.echo(f"✗ Failed to process: {failed} subjects", err=True)

app.command()(batch)

if __name__ == "__main__":
    app()