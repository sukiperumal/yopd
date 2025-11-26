#!/usr/bin/env python3

"""
Functional Graph Construction from fMRI Data

This module constructs functional brain graphs from fMRI data processed by fMRIPrep.
It extracts BOLD time series from atlas regions and computes functional connectivity
to create brain networks.

Key components:
- Extract ROI time series from preprocessed BOLD data
- Compute functional connectivity matrices (correlation-based)
- Apply thresholding to create sparse networks
- Calculate graph-theoretic metrics for each node
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from dataclasses import dataclass
from nilearn import datasets, image, maskers, connectome, plotting
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionalConfig:
    """Configuration for functional graph construction"""
    schaefer_n_rois: int = 200
    correlation_threshold: float = 0.3
    connectivity_metric: str = 'correlation'  # 'correlation', 'partial correlation', 'covariance'
    detrend: bool = True
    standardize: bool = True
    low_pass: Optional[float] = 0.1  # Hz
    high_pass: Optional[float] = 0.01  # Hz
    tr: Optional[float] = None  # Will be extracted from data if None
    confounds_strategy: List[str] = None
    
    def __post_init__(self):
        if self.confounds_strategy is None:
            self.confounds_strategy = ['motion', 'wm_csf', 'global_signal']

class FMRIPrepDataLoader:
    """Load and preprocess BOLD data from fMRIPrep output"""
    
    def __init__(self, fmriprep_subject_dir: Path, session: str = None):
        self.subject_dir = fmriprep_subject_dir
        self.session = session
        self.subject_id = fmriprep_subject_dir.name  # Should be 'sub-XXXXX'
        self._validate_subject_dir()
    
    def _validate_subject_dir(self):
        """Validate fMRIPrep subject directory structure"""
        if not self.subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {self.subject_dir}")
        
        if not self.subject_id.startswith('sub-'):
            raise ValueError(f"Invalid subject directory name: {self.subject_id}")
        
        logger.info(f"Loading data for {self.subject_id}")
    
    def find_bold_files(self, task: str = None, space: str = 'MNI152NLin2009cAsym') -> List[Path]:
        """Find preprocessed BOLD files in fMRIPrep output"""
        try:
            # Search for BOLD files
            search_pattern = f"*_space-{space}_desc-preproc_bold.nii*"
            
            if self.session:
                func_dir = self.subject_dir / f"ses-{self.session}" / "func"
            else:
                # Search all sessions
                func_dirs = list(self.subject_dir.glob("ses-*/func"))
                if not func_dirs:
                    func_dirs = [self.subject_dir / "func"]  # No session structure
            
            bold_files = []
            
            if self.session:
                func_dirs = [func_dir]
            
            for func_dir in func_dirs:
                if func_dir.exists():
                    files = list(func_dir.glob(search_pattern))
                    
                    # Filter by task if specified
                    if task:
                        files = [f for f in files if f"task-{task}" in f.name]
                    
                    bold_files.extend(files)
            
            if not bold_files:
                # Try to find any BOLD files with different patterns
                alt_patterns = [
                    "*bold*.nii*",
                    f"*_space-{space}_*bold*.nii*",
                    "*_desc-preproc_*bold*.nii*"
                ]
                
                for pattern in alt_patterns:
                    for func_dir in func_dirs:
                        if func_dir.exists():
                            alt_files = list(func_dir.glob(pattern))
                            if alt_files:
                                logger.warning(f"Found BOLD files with alternative pattern: {pattern}")
                                bold_files.extend(alt_files)
                                break
                    if bold_files:
                        break
            
            logger.info(f"Found {len(bold_files)} BOLD files")
            for bf in bold_files:
                logger.info(f"  - {bf.name}")
            
            return bold_files
            
        except Exception as e:
            logger.error(f"Failed to find BOLD files: {e}")
            return []
    
    def find_confounds_file(self, bold_file: Path) -> Optional[Path]:
        """Find corresponding confounds file for a BOLD file"""
        try:
            # Replace desc-preproc_bold with desc-confounds_timeseries
            confounds_name = bold_file.name.replace(
                '_desc-preproc_bold.nii.gz', 
                '_desc-confounds_timeseries.tsv'
            ).replace(
                '_desc-preproc_bold.nii', 
                '_desc-confounds_timeseries.tsv'
            ).replace(
                '_bold.nii.gz',
                '_desc-confounds_timeseries.tsv'  
            ).replace(
                '_bold.nii',
                '_desc-confounds_timeseries.tsv'
            )
            
            confounds_file = bold_file.parent / confounds_name
            
            if confounds_file.exists():
                return confounds_file
            else:
                logger.warning(f"Confounds file not found: {confounds_file}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to find confounds file: {e}")
            return None
    
    def load_confounds(self, confounds_file: Path, strategy: List[str]) -> Optional[pd.DataFrame]:
        """Load and select confound regressors"""
        try:
            if not confounds_file or not confounds_file.exists():
                logger.warning("No confounds file available")
                return None
            
            # Load confounds
            confounds_df = pd.read_csv(confounds_file, sep='\t')
            
            # Select confounds based on strategy
            selected_confounds = []
            
            for strategy_item in strategy:
                if strategy_item == 'motion':
                    # Standard motion parameters
                    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
                    available_motion = [col for col in motion_cols if col in confounds_df.columns]
                    selected_confounds.extend(available_motion)
                
                elif strategy_item == 'motion_derivatives':
                    # Motion derivatives
                    motion_deriv_cols = [col for col in confounds_df.columns 
                                       if '_derivative1' in col and any(motion in col for motion in ['trans', 'rot'])]
                    selected_confounds.extend(motion_deriv_cols)
                
                elif strategy_item == 'wm_csf':
                    # White matter and CSF signals
                    wm_csf_cols = ['white_matter', 'csf']
                    available_wm_csf = [col for col in wm_csf_cols if col in confounds_df.columns]
                    selected_confounds.extend(available_wm_csf)
                
                elif strategy_item == 'global_signal':
                    # Global signal
                    if 'global_signal' in confounds_df.columns:
                        selected_confounds.append('global_signal')
                
                elif strategy_item == 'compcor':
                    # CompCor components
                    compcor_cols = [col for col in confounds_df.columns if 'comp_cor' in col]
                    selected_confounds.extend(compcor_cols[:6])  # First 6 components
            
            if selected_confounds:
                confounds_subset = confounds_df[selected_confounds]
                
                # Handle missing values
                confounds_subset = confounds_subset.fillna(0)
                
                logger.info(f"Selected {len(selected_confounds)} confound regressors")
                return confounds_subset
            else:
                logger.warning("No confound regressors found")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load confounds: {e}")
            return None
    
    def extract_tr_from_json(self, bold_file: Path) -> Optional[float]:
        """Extract TR (repetition time) from BOLD JSON sidecar"""
        try:
            json_file = bold_file.with_suffix('').with_suffix('.json')
            
            if json_file.exists():
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                tr = metadata.get('RepetitionTime')
                if tr:
                    logger.info(f"Extracted TR: {tr} seconds")
                    return float(tr)
            
            logger.warning(f"Could not extract TR from {json_file}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract TR: {e}")
            return None

class FunctionalAtlasExtractor:
    """Extract time series from atlas-defined ROIs"""
    
    def __init__(self, config: FunctionalConfig):
        self.config = config
        self.atlas_data = None
        self.labels = None
        self.masker = None
        self._load_atlas()
    
    def _load_atlas(self):
        """Load Schaefer atlas"""
        try:
            atlas = datasets.fetch_atlas_schaefer_2018(
                n_rois=self.config.schaefer_n_rois,
                yeo_networks=7,
                resolution_mm=2
            )
            
            self.atlas_data = atlas['maps']
            self.labels = [label.decode() if isinstance(label, bytes) else label 
                          for label in atlas['labels']]
            
            # Create masker
            self.masker = NiftiLabelsMasker(
                labels_img=self.atlas_data,
                labels=self.labels,
                standardize=self.config.standardize,
                detrend=self.config.detrend,
                low_pass=self.config.low_pass,
                high_pass=self.config.high_pass,
                t_r=self.config.tr
            )
            
            logger.info(f"Loaded Schaefer atlas with {self.config.schaefer_n_rois} ROIs")
            
        except Exception as e:
            logger.error(f"Failed to load atlas: {e}")
            raise
    
    def extract_time_series(self, bold_file: Path, confounds: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Extract ROI time series from BOLD data"""
        try:
            # Load BOLD image
            bold_img = nib.load(bold_file)
            logger.info(f"Loaded BOLD image: {bold_img.shape}")
            
            # Update masker with TR if available
            if self.config.tr is None:
                # Try to extract TR from the data
                loader = FMRIPrepDataLoader(bold_file.parent.parent.parent)
                tr = loader.extract_tr_from_json(bold_file)
                if tr:
                    self.masker.t_r = tr
            
            # Prepare confounds
            confounds_array = None
            if confounds is not None:
                confounds_array = confounds.values
                logger.info(f"Using {confounds_array.shape[1]} confound regressors")
            
            # Extract time series
            time_series = self.masker.fit_transform(bold_img, confounds=confounds_array)
            
            logger.info(f"Extracted time series: {time_series.shape}")
            return time_series
            
        except Exception as e:
            logger.error(f"Failed to extract time series from {bold_file}: {e}")
            raise

class FunctionalConnectivityAnalyzer:
    """Compute functional connectivity and graph metrics"""
    
    def __init__(self, config: FunctionalConfig):
        self.config = config
    
    def compute_connectivity_matrix(self, time_series: np.ndarray) -> np.ndarray:
        """Compute functional connectivity matrix"""
        try:
            connectivity_measure = ConnectivityMeasure(
                kind=self.config.connectivity_metric,
                discard_diagonal=True
            )
            
            connectivity_matrix = connectivity_measure.fit_transform([time_series])[0]
            
            logger.info(f"Computed {self.config.connectivity_metric} connectivity matrix: {connectivity_matrix.shape}")
            return connectivity_matrix
            
        except Exception as e:
            logger.error(f"Failed to compute connectivity matrix: {e}")
            raise
    
    def threshold_connectivity(self, connectivity_matrix: np.ndarray) -> np.ndarray:
        """Apply threshold to create sparse connectivity matrix"""
        try:
            if self.config.connectivity_metric == 'correlation':
                # For correlation, threshold positive correlations
                thresholded_matrix = np.where(
                    connectivity_matrix > self.config.correlation_threshold,
                    connectivity_matrix,
                    0
                )
            else:
                # For other metrics, use absolute threshold
                thresholded_matrix = np.where(
                    np.abs(connectivity_matrix) > self.config.correlation_threshold,
                    connectivity_matrix,
                    0
                )
            
            # Ensure diagonal is zero
            np.fill_diagonal(thresholded_matrix, 0)
            
            n_edges = np.sum(thresholded_matrix != 0) // 2
            density = n_edges / (thresholded_matrix.shape[0] * (thresholded_matrix.shape[0] - 1) / 2)
            
            logger.info(f"Thresholded connectivity: {n_edges} edges, density = {density:.3f}")
            return thresholded_matrix
            
        except Exception as e:
            logger.error(f"Failed to threshold connectivity: {e}")
            raise
    
    def compute_graph_metrics(self, adjacency_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute graph-theoretic metrics"""
        try:
            # Create NetworkX graph (use absolute values for graph metrics)
            adj_for_graph = np.abs(adjacency_matrix)
            G = nx.from_numpy_array(adj_for_graph)
            
            metrics = {}
            
            # Degree centrality
            metrics['degree'] = np.array(list(dict(G.degree(weight='weight')).values()))
            
            # Strength (weighted degree)
            metrics['strength'] = np.array([sum([G[node][neighbor]['weight'] 
                                               for neighbor in G.neighbors(node)]) 
                                          for node in G.nodes()])
            
            # Betweenness centrality
            metrics['betweenness'] = np.array(list(
                nx.betweenness_centrality(G, weight='weight').values()
            ))
            
            # Closeness centrality  
            metrics['closeness'] = np.array(list(
                nx.closeness_centrality(G, distance='weight').values()
            ))
            
            # Clustering coefficient
            metrics['clustering'] = np.array(list(
                nx.clustering(G, weight='weight').values()
            ))
            
            # Local efficiency
            metrics['local_efficiency'] = np.array([
                nx.local_efficiency(G, node) for node in G.nodes()
            ])
            
            # Eigenvector centrality (if graph is connected)
            try:
                metrics['eigenvector'] = np.array(list(
                    nx.eigenvector_centrality(G, weight='weight', max_iter=1000).values()
                ))
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                logger.warning("Eigenvector centrality computation failed, using zeros")
                metrics['eigenvector'] = np.zeros(len(G.nodes()))
            
            logger.info(f"Computed graph metrics for {len(G.nodes())} nodes")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute graph metrics: {e}")
            # Return empty metrics as fallback
            n_nodes = adjacency_matrix.shape[0]
            return {metric: np.zeros(n_nodes) for metric in 
                   ['degree', 'strength', 'betweenness', 'closeness', 'clustering', 
                    'local_efficiency', 'eigenvector']}

class FunctionalGraphConstructor:
    """Main class for constructing functional brain graphs"""
    
    def __init__(self, config: FunctionalConfig = None):
        self.config = config or FunctionalConfig()
        self.atlas_extractor = FunctionalAtlasExtractor(self.config)
        self.connectivity_analyzer = FunctionalConnectivityAnalyzer(self.config)
    
    def construct_functional_graph(self, fmriprep_subject_dir: Path, 
                                 task: str = None, 
                                 session: str = None,
                                 run: str = None) -> Dict:
        """
        Main method to construct functional graph from fMRIPrep output
        
        Args:
            fmriprep_subject_dir: Path to subject directory in fMRIPrep derivatives
                                 (e.g., derivatives/fmriprep/sub-01)
            task: Task name to process (e.g., 'rest', 'task')
            session: Session identifier (e.g., '01')
            run: Run identifier (e.g., '1')
        
        Returns:
            Dictionary containing functional graph data
        """
        try:
            logger.info(f"Constructing functional graph for: {fmriprep_subject_dir}")
            
            # Load fMRIPrep data
            data_loader = FMRIPrepDataLoader(fmriprep_subject_dir, session)
            
            # Find BOLD files
            bold_files = data_loader.find_bold_files(task=task)
            
            if not bold_files:
                raise FileNotFoundError(f"No BOLD files found for subject {fmriprep_subject_dir.name}")
            
            # Process the first BOLD file (or filter by run if specified)
            bold_file = bold_files[0]
            if run and len(bold_files) > 1:
                run_files = [f for f in bold_files if f"run-{run}" in f.name]
                if run_files:
                    bold_file = run_files[0]
                else:
                    logger.warning(f"Run {run} not found, using first available: {bold_file.name}")
            
            logger.info(f"Processing BOLD file: {bold_file.name}")
            
            # Extract TR if not provided
            if self.config.tr is None:
                tr = data_loader.extract_tr_from_json(bold_file)
                if tr:
                    self.config.tr = tr
                    self.atlas_extractor.masker.t_r = tr
            
            # Load confounds
            confounds_file = data_loader.find_confounds_file(bold_file)
            confounds = data_loader.load_confounds(confounds_file, self.config.confounds_strategy)
            
            # Extract time series
            time_series = self.atlas_extractor.extract_time_series(bold_file, confounds)
            
            # Compute connectivity matrix
            connectivity_matrix = self.connectivity_analyzer.compute_connectivity_matrix(time_series)
            
            # Apply thresholding
            thresholded_matrix = self.connectivity_analyzer.threshold_connectivity(connectivity_matrix)
            
            # Compute graph metrics
            graph_metrics = self.connectivity_analyzer.compute_graph_metrics(thresholded_matrix)
            
            # Create NetworkX graph
            G = nx.from_numpy_array(np.abs(thresholded_matrix))
            
            # Add node attributes
            for i, label in enumerate(self.atlas_extractor.labels):
                G.nodes[i]['label'] = label
                G.nodes[i]['time_series_mean'] = float(np.mean(time_series[:, i]))
                G.nodes[i]['time_series_std'] = float(np.std(time_series[:, i]))
                
                # Add graph metrics as node attributes
                for metric_name, metric_values in graph_metrics.items():
                    G.nodes[i][metric_name] = float(metric_values[i])
            
            # Prepare output
            functional_graph = {
                'adjacency_matrix': thresholded_matrix,
                'connectivity_matrix': connectivity_matrix,
                'time_series': time_series,
                'graph_metrics': graph_metrics,
                'functional_network': G,
                'roi_labels': self.atlas_extractor.labels,
                'n_rois': self.config.schaefer_n_rois,
                'threshold': self.config.correlation_threshold,
                'connectivity_metric': self.config.connectivity_metric,
                'tr': self.config.tr,
                'confounds_used': confounds is not None,
                'n_timepoints': time_series.shape[0],
                'bold_file': str(bold_file),
                'config': self.config
            }
            
            logger.info("Functional graph construction completed successfully")
            return functional_graph
            
        except Exception as e:
            logger.error(f"Functional graph construction failed: {e}")
            raise

def construct_functional_graph_cli(subject_dir: str,
                                 output_file: str = None,
                                 task: str = None,
                                 session: str = None, 
                                 run: str = None,
                                 n_rois: int = 200,
                                 correlation_threshold: float = 0.3,
                                 connectivity_metric: str = 'correlation') -> str:
    """
    Command-line interface for functional graph construction
    
    Args:
        subject_dir: Path to fMRIPrep subject directory
        output_file: Output file path (optional)
        task: Task name to process
        session: Session identifier
        run: Run identifier
        n_rois: Number of ROIs in Schaefer atlas
        correlation_threshold: Threshold for functional connectivity
        connectivity_metric: Type of connectivity measure
    
    Returns:
        Path to output file
    """
    try:
        # Configure functional graph construction
        config = FunctionalConfig(
            schaefer_n_rois=n_rois,
            correlation_threshold=correlation_threshold,
            connectivity_metric=connectivity_metric
        )
        
        # Construct graph
        constructor = FunctionalGraphConstructor(config)
        functional_graph = constructor.construct_functional_graph(
            Path(subject_dir), 
            task=task,
            session=session,
            run=run
        )
        
        # Determine output file
        if output_file is None:
            subject_name = Path(subject_dir).name
            output_file = f"{subject_name}_functional_graph.pkl"
        
        # Save graph
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(functional_graph, f)
        
        logger.info(f"Functional graph saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"CLI construction failed: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        subject_dir = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        task = sys.argv[3] if len(sys.argv) > 3 else None
        
        result_file = construct_functional_graph_cli(
            subject_dir=subject_dir,
            output_file=output_file,
            task=task
        )
        
        print(f"Functional graph constructed and saved to: {result_file}")
    else:
        print("Usage: python graph_fmri.py <subject_dir> [output_file] [task]")
        print("Example: python graph_fmri.py derivatives/fmriprep/sub-01 sub-01_func.pkl rest")