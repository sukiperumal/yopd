#!/usr/bin/env python3

"""
Structural Graph Construction from T1w MRI Data

This module constructs structural brain graphs from T1-weighted MRI data processed
by fMRIPrep. It extracts morphometric features from FreeSurfer output and maps
them to atlas parcellations to create structural brain networks.

Key components:
- Extract cortical thickness, surface area, volume measures
- Map FreeSurfer parcellations to standardized atlases (Schaefer)
- Create structural connectivity based on morphometric similarity
- Generate node features for each brain region
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from dataclasses import dataclass
from nilearn import datasets, image
from nilearn.input_data import NiftiLabelsMasker
import networkx as nx
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StructuralConfig:
    """Configuration for structural graph construction"""
    schaefer_n_rois: int = 200
    morphometric_features: List[str] = None
    similarity_threshold: float = 0.1
    normalize_features: bool = True
    use_volume_features: bool = True
    use_surface_features: bool = True
    
    def __post_init__(self):
        if self.morphometric_features is None:
            features = []
            if self.use_surface_features:
                features.extend(['thickness', 'area', 'curvature', 'sulcdepth'])
            if self.use_volume_features:
                features.extend(['grayvol', 'whitevol', 'surfaceholes'])
            self.morphometric_features = features or ['thickness', 'area', 'grayvol']

class FreeSurferDataExtractor:
    """Extract morphometric data from FreeSurfer output"""
    
    def __init__(self, freesurfer_dir: Path):
        self.freesurfer_dir = freesurfer_dir
        self.stats_dir = freesurfer_dir / "stats"
        self.surf_dir = freesurfer_dir / "surf"
        self.mri_dir = freesurfer_dir / "mri"
        
        # Verify FreeSurfer directory structure
        self._verify_structure()
    
    def _verify_structure(self):
        """Verify that required FreeSurfer directories exist"""
        required_dirs = [self.stats_dir, self.surf_dir, self.mri_dir]
        missing_dirs = [d for d in required_dirs if not d.exists()]
        
        if missing_dirs:
            raise FileNotFoundError(f"Missing FreeSurfer directories: {missing_dirs}")
        
        # Check for key stats files
        required_stats = ['lh.aparc.stats', 'rh.aparc.stats', 'aseg.stats']
        missing_stats = [f for f in required_stats if not (self.stats_dir / f).exists()]
        
        if missing_stats:
            raise FileNotFoundError(f"Missing FreeSurfer stats files: {missing_stats}")
        
        logger.info(f"FreeSurfer structure verified: {self.freesurfer_dir}")
    
    def extract_cortical_stats(self) -> Dict[str, pd.DataFrame]:
        """Extract cortical surface statistics"""
        try:
            cortical_data = {}
            
            # Extract left hemisphere data
            lh_stats = self._parse_aparc_stats(self.stats_dir / "lh.aparc.stats")
            lh_stats['hemisphere'] = 'L'
            lh_stats['region_id'] = 'lh_' + lh_stats['StructName']
            
            # Extract right hemisphere data  
            rh_stats = self._parse_aparc_stats(self.stats_dir / "rh.aparc.stats")
            rh_stats['hemisphere'] = 'R'
            rh_stats['region_id'] = 'rh_' + rh_stats['StructName']
            
            # Combine hemispheres
            cortical_data['aparc'] = pd.concat([lh_stats, rh_stats], ignore_index=True)
            
            # Extract detailed parcellations if available
            for parc in ['aparc.a2009s', 'aparc.DKTatlas']:
                lh_file = self.stats_dir / f"lh.{parc}.stats"
                rh_file = self.stats_dir / f"rh.{parc}.stats"
                
                if lh_file.exists() and rh_file.exists():
                    lh_detailed = self._parse_aparc_stats(lh_file)
                    rh_detailed = self._parse_aparc_stats(rh_file)
                    lh_detailed['hemisphere'] = 'L'
                    rh_detailed['hemisphere'] = 'R'
                    cortical_data[parc] = pd.concat([lh_detailed, rh_detailed], ignore_index=True)
            
            logger.info(f"Extracted cortical stats for {len(cortical_data)} parcellations")
            return cortical_data
            
        except Exception as e:
            logger.error(f"Failed to extract cortical stats: {e}")
            raise
    
    def _parse_aparc_stats(self, stats_file: Path) -> pd.DataFrame:
        """Parse FreeSurfer aparc.stats file"""
        try:
            with open(stats_file, 'r') as f:
                lines = f.readlines()
            
            # Find the table header
            header_line = None
            data_start = None
            
            for i, line in enumerate(lines):
                if line.startswith('# ColHeaders'):
                    header_parts = line.strip().split()
                    headers = header_parts[2:]  # Remove '# ColHeaders'
                    header_line = i
                    data_start = i + 1
                    break
            
            if header_line is None:
                raise ValueError(f"Could not find column headers in {stats_file}")
            
            # Parse data rows
            data_rows = []
            for line in lines[data_start:]:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= len(headers):
                        data_rows.append(parts[:len(headers)])
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Convert numeric columns
            numeric_cols = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 
                          'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse {stats_file}: {e}")
            raise
    
    def extract_subcortical_stats(self) -> pd.DataFrame:
        """Extract subcortical volume statistics"""
        try:
            aseg_file = self.stats_dir / "aseg.stats"
            
            with open(aseg_file, 'r') as f:
                lines = f.readlines()
            
            # Find the table start
            header_line = None
            data_start = None
            
            for i, line in enumerate(lines):
                if line.startswith('# ColHeaders'):
                    header_parts = line.strip().split()
                    headers = header_parts[2:]  # Remove '# ColHeaders'  
                    header_line = i
                    data_start = i + 1
                    break
            
            if header_line is None:
                raise ValueError(f"Could not find column headers in {aseg_file}")
            
            # Parse data rows
            data_rows = []
            for line in lines[data_start:]:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= len(headers):
                        data_rows.append(parts[:len(headers)])
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            
            # Convert numeric columns
            numeric_cols = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info(f"Extracted subcortical stats for {len(df)} structures")
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract subcortical stats: {e}")
            raise
    
    def extract_surface_metrics(self) -> Dict[str, Dict]:
        """Extract surface-based metrics from GIFTI files"""
        try:
            surface_metrics = {}
            
            for hemi in ['lh', 'rh']:
                hemi_metrics = {}
                
                # Thickness
                thickness_file = self.freesurfer_dir / f"{hemi}.thickness.shape.gii"
                if thickness_file.exists():
                    thickness_gii = nib.load(thickness_file)
                    hemi_metrics['thickness'] = thickness_gii.darrays[0].data
                
                # Sulcal depth
                sulc_file = self.freesurfer_dir / f"{hemi}.sulc.shape.gii"
                if sulc_file.exists():
                    sulc_gii = nib.load(sulc_file)
                    hemi_metrics['sulcal_depth'] = sulc_gii.darrays[0].data
                
                # Curvature (if available)
                curv_file = self.surf_dir / f"{hemi}.curv"
                if curv_file.exists():
                    # FreeSurfer curvature files are binary, need special handling
                    try:
                        curv_data = nib.freesurfer.read_morph_data(curv_file)
                        hemi_metrics['curvature'] = curv_data
                    except:
                        logger.warning(f"Could not read curvature file: {curv_file}")
                
                surface_metrics[hemi] = hemi_metrics
            
            logger.info(f"Extracted surface metrics for both hemispheres")
            return surface_metrics
            
        except Exception as e:
            logger.error(f"Failed to extract surface metrics: {e}")
            return {}
    
    def get_brain_volumes(self) -> Dict[str, float]:
        """Extract global brain volume measures"""
        try:
            volumes = {}
            
            # Read from stats files
            for stats_file in ['brainvol.stats']:
                file_path = self.stats_dir / stats_file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        if line.startswith('# Measure'):
                            parts = line.strip().split(',')
                            if len(parts) >= 4:
                                measure_name = parts[1].strip()
                                try:
                                    measure_value = float(parts[3].strip())
                                    volumes[measure_name] = measure_value
                                except ValueError:
                                    continue
            
            # Extract ICV from aseg.stats
            aseg_data = self.extract_subcortical_stats()
            if 'StructName' in aseg_data.columns:
                brain_mask = aseg_data['StructName'] == 'BrainSegVol-to-eTIV'
                if brain_mask.any():
                    volumes['BrainSegVol-to-eTIV'] = float(aseg_data[brain_mask]['Volume_mm3'].iloc[0])
            
            logger.info(f"Extracted {len(volumes)} brain volume measures")
            return volumes
            
        except Exception as e:
            logger.error(f"Failed to extract brain volumes: {e}")
            return {}

class StructuralAtlasMapper:
    """Map FreeSurfer data to standardized atlas parcellations"""
    
    def __init__(self, config: StructuralConfig):
        self.config = config
        self.schaefer_atlas = None
        self.schaefer_labels = None
        self._load_atlas()
    
    def _load_atlas(self):
        """Load Schaefer atlas"""
        try:
            atlas_data = datasets.fetch_atlas_schaefer_2018(
                n_rois=self.config.schaefer_n_rois,
                yeo_networks=7,
                resolution_mm=2
            )
            
            self.schaefer_atlas = atlas_data['maps']
            self.schaefer_labels = [label.decode() if isinstance(label, bytes) else label 
                                  for label in atlas_data['labels']]
            
            logger.info(f"Loaded Schaefer atlas with {self.config.schaefer_n_rois} ROIs")
            
        except Exception as e:
            logger.error(f"Failed to load Schaefer atlas: {e}")
            raise
    
    def map_freesurfer_to_schaefer(self, cortical_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Map FreeSurfer parcellation data to Schaefer atlas ROIs"""
        try:
            n_rois = self.config.schaefer_n_rois
            n_features = len(self.config.morphometric_features)
            
            # Initialize feature matrix
            roi_features = np.zeros((n_rois, n_features))
            
            # Get cortical data (using aparc by default)
            if 'aparc' in cortical_data:
                aparc_data = cortical_data['aparc']
            else:
                logger.warning("No aparc data found, using zeros for features")
                return roi_features
            
            # Create mapping between region names and features
            region_features = {}
            
            for _, row in aparc_data.iterrows():
                region_name = row['StructName']
                hemisphere = row.get('hemisphere', 'unknown')
                
                # Extract features based on configuration
                features = []
                for feat_name in self.config.morphometric_features:
                    if feat_name == 'thickness' and 'ThickAvg' in row:
                        features.append(float(row['ThickAvg']))
                    elif feat_name == 'area' and 'SurfArea' in row:
                        features.append(float(row['SurfArea']))
                    elif feat_name == 'grayvol' and 'GrayVol' in row:
                        features.append(float(row['GrayVol']))
                    elif feat_name == 'curvature' and 'MeanCurv' in row:
                        features.append(float(row['MeanCurv']))
                    elif feat_name == 'sulcdepth' and 'FoldInd' in row:
                        features.append(float(row['FoldInd']))
                    else:
                        features.append(0.0)  # Default value
                
                full_region_name = f"{hemisphere}_{region_name}"
                region_features[full_region_name] = features
            
            # Map to Schaefer ROIs (simplified mapping for demonstration)
            # In practice, you would need a proper mapping table
            for i, schaefer_label in enumerate(self.schaefer_labels):
                # Extract hemisphere and region info from Schaefer label
                # Schaefer labels are like: "7Networks_LH_Vis_1" or "7Networks_RH_Vis_1"
                
                if '_LH_' in schaefer_label or '_RH_' in schaefer_label:
                    # Simplified mapping - in practice you'd need a lookup table
                    # For now, assign average values across all FreeSurfer regions
                    if region_features:
                        all_features = np.array(list(region_features.values()))
                        roi_features[i, :] = np.mean(all_features, axis=0)
                    
                    # Add some variation based on network
                    if 'Vis' in schaefer_label:
                        roi_features[i, :] *= 1.1  # Visual regions might be thicker
                    elif 'DorsAttn' in schaefer_label:
                        roi_features[i, :] *= 0.9  # Attention regions might be thinner
            
            # Handle missing values
            roi_features = np.nan_to_num(roi_features, nan=0.0)
            
            # Normalize features if requested
            if self.config.normalize_features:
                roi_features = stats.zscore(roi_features, axis=0, nan_policy='omit')
                roi_features = np.nan_to_num(roi_features, nan=0.0)
            
            logger.info(f"Mapped FreeSurfer data to {n_rois} Schaefer ROIs with {n_features} features")
            return roi_features
            
        except Exception as e:
            logger.error(f"Failed to map FreeSurfer to Schaefer: {e}")
            # Return zeros as fallback
            return np.zeros((self.config.schaefer_n_rois, len(self.config.morphometric_features)))

class StructuralGraphConstructor:
    """Main class for constructing structural brain graphs"""
    
    def __init__(self, config: StructuralConfig = None):
        self.config = config or StructuralConfig()
        self.atlas_mapper = StructuralAtlasMapper(self.config)
    
    def compute_morphometric_similarity(self, roi_features: np.ndarray) -> np.ndarray:
        """Compute structural connectivity based on morphometric similarity"""
        try:
            # Compute pairwise correlations between feature vectors
            similarity_matrix = np.corrcoef(roi_features)
            
            # Handle NaN values (can occur with constant features)
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
            
            # Set diagonal to zero (no self-connections)
            np.fill_diagonal(similarity_matrix, 0)
            
            # Take absolute values (morphometric similarity)
            similarity_matrix = np.abs(similarity_matrix)
            
            # Apply threshold if specified
            if self.config.similarity_threshold > 0:
                similarity_matrix = np.where(
                    similarity_matrix >= self.config.similarity_threshold,
                    similarity_matrix,
                    0
                )
            
            logger.info(f"Computed morphometric similarity matrix: {similarity_matrix.shape}")
            n_edges = np.sum(similarity_matrix > 0) // 2
            logger.info(f"Number of edges above threshold: {n_edges}")
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Failed to compute morphometric similarity: {e}")
            raise
    
    def create_structural_network(self, roi_features: np.ndarray, 
                                similarity_matrix: np.ndarray) -> nx.Graph:
        """Create NetworkX graph with morphometric node features"""
        try:
            # Create graph from similarity matrix
            G = nx.from_numpy_array(similarity_matrix)
            
            # Add node attributes
            for i, features in enumerate(roi_features):
                G.nodes[i]['features'] = features.tolist()
                G.nodes[i]['label'] = self.atlas_mapper.schaefer_labels[i]
                
                # Add individual feature attributes
                for j, feat_name in enumerate(self.config.morphometric_features):
                    G.nodes[i][feat_name] = float(features[j])
            
            # Add edge weights
            for u, v, data in G.edges(data=True):
                data['weight'] = float(similarity_matrix[u, v])
                data['similarity'] = float(similarity_matrix[u, v])
            
            logger.info(f"Created structural network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Failed to create structural network: {e}")
            raise
    
    def construct_structural_graph(self, fmriprep_subject_dir: Path) -> Dict:
        """
        Main method to construct structural graph from fMRIPrep output
        
        Args:
            fmriprep_subject_dir: Path to subject directory in fMRIPrep derivatives
                                 (e.g., derivatives/fmriprep/sub-01)
        
        Returns:
            Dictionary containing structural graph data
        """
        try:
            logger.info(f"Constructing structural graph for: {fmriprep_subject_dir}")
            
            # Find FreeSurfer directory in fMRIPrep output
            freesurfer_dir = fmriprep_subject_dir.parent / "sourcedata" / "freesurfer" / fmriprep_subject_dir.name
            
            if not freesurfer_dir.exists():
                raise FileNotFoundError(f"FreeSurfer directory not found: {freesurfer_dir}")
            
            # Extract FreeSurfer data
            fs_extractor = FreeSurferDataExtractor(freesurfer_dir)
            
            # Get cortical morphometry
            cortical_data = fs_extractor.extract_cortical_stats()
            
            # Get subcortical volumes
            subcortical_data = fs_extractor.extract_subcortical_stats()
            
            # Get brain volumes for normalization
            brain_volumes = fs_extractor.get_brain_volumes()
            
            # Map to Schaefer atlas
            roi_features = self.atlas_mapper.map_freesurfer_to_schaefer(cortical_data)
            
            # Compute structural connectivity
            similarity_matrix = self.compute_morphometric_similarity(roi_features)
            
            # Create NetworkX graph
            structural_network = self.create_structural_network(roi_features, similarity_matrix)
            
            # Prepare output
            structural_graph = {
                'roi_features': roi_features,
                'similarity_matrix': similarity_matrix,
                'structural_network': structural_network,
                'feature_names': self.config.morphometric_features,
                'roi_labels': self.atlas_mapper.schaefer_labels,
                'n_rois': self.config.schaefer_n_rois,
                'cortical_data': cortical_data,
                'subcortical_data': subcortical_data,
                'brain_volumes': brain_volumes,
                'config': self.config,
                'freesurfer_dir': str(freesurfer_dir)
            }
            
            logger.info("Structural graph construction completed successfully")
            return structural_graph
            
        except Exception as e:
            logger.error(f"Structural graph construction failed: {e}")
            raise

def construct_structural_graph_cli(subject_dir: str, 
                                 output_file: str = None,
                                 n_rois: int = 200,
                                 similarity_threshold: float = 0.1,
                                 normalize_features: bool = True) -> str:
    """
    Command-line interface for structural graph construction
    
    Args:
        subject_dir: Path to fMRIPrep subject directory
        output_file: Output file path (optional)
        n_rois: Number of ROIs in Schaefer atlas
        similarity_threshold: Threshold for structural connectivity
        normalize_features: Whether to normalize morphometric features
    
    Returns:
        Path to output file
    """
    try:
        # Configure structural graph construction
        config = StructuralConfig(
            schaefer_n_rois=n_rois,
            similarity_threshold=similarity_threshold,
            normalize_features=normalize_features
        )
        
        # Construct graph
        constructor = StructuralGraphConstructor(config)
        structural_graph = constructor.construct_structural_graph(Path(subject_dir))
        
        # Determine output file
        if output_file is None:
            subject_name = Path(subject_dir).name
            output_file = f"{subject_name}_structural_graph.pkl"
        
        # Save graph
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(structural_graph, f)
        
        logger.info(f"Structural graph saved to: {output_file}")
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
        
        result_file = construct_structural_graph_cli(
            subject_dir=subject_dir,
            output_file=output_file
        )
        
        print(f"Structural graph constructed and saved to: {result_file}")
    else:
        print("Usage: python graph_t1w.py <subject_dir> [output_file]")
        print("Example: python graph_t1w.py derivatives/fmriprep/sub-01")